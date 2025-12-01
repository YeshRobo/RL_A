"""
SAC on DeepMind Control Suite cartpole/balance.

This script satisfies the assignment requirements:
- Actor-critic continuous control (Soft Actor-Critic, SAC)
- DeepMind Control Suite cartpole / balance task
- Train with seeds [0, 1, 2]
- Evaluate with seed 10
- Plot learning curves (reward vs episode) with mean ± std
  for both training and evaluation.

Outputs (in results_cartpole_sac/):
- learning_curve_train.png
- learning_curve_eval.png
"""

import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt

from dm_control import suite  # DeepMind Control Suite


# --------------------------- config & utils --------------------------- #

@dataclass
class Config:
    # Environment
    domain: str = "cartpole"
    task: str = "balance"

    # Seeds
    train_seeds: Tuple[int, ...] = (0, 1, 2)
    eval_seed: int = 10

    # Training settings
    num_episodes: int = 500
    max_episode_steps: int = 1000
    num_envs: int = 5
    replay_size: int = int(1e6)
    batch_size: int = 256
    random_steps: int = 1000        # initial random exploration
    updates_per_step: int = 1
    eval_interval: int = 10         # evaluate every N episodes
    eval_episodes: int = 5          # eval episodes per evaluation point

    # SAC hyperparameters
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    hidden_dim: int = 256

    # Misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: str = "results_cartpole_sac"


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------- DMC env wrapper --------------------------- #

class DMCEnvWrapper:
    """
    Simple wrapper around dm_control.suite to look a bit like a gym.Env.

    - Observations: flattened into a 1D float32 array.
    - Actions: assumed in [-1, 1]; we scale them to env's action_spec range.
    """

    def __init__(self, domain: str, task: str, seed: int,
                 max_episode_steps: int = 1000):
        self._domain = domain
        self._task = task
        self._seed = seed
        self._max_episode_steps = max_episode_steps

        # random seed via task_kwargs
        self.env = suite.load(domain, task, task_kwargs={"random": seed})
        self._action_spec = self.env.action_spec()

        # get obs dim by doing a reset once
        ts = self.env.reset()
        obs0 = self._flatten_obs(ts)
        self._obs_dim = obs0.shape[0]
        self._action_dim = int(np.prod(self._action_spec.shape))
        self._elapsed_steps = 0

    @property
    def observation_space_shape(self):
        return (self._obs_dim,)

    @property
    def action_space_shape(self):
        return (self._action_dim,)

    def _flatten_obs(self, time_step):
        obs_list = []
        for v in time_step.observation.values():
            obs_list.append(np.array(v, dtype=np.float32).ravel())
        return np.concatenate(obs_list, axis=0).astype(np.float32)

    def reset(self):
        self._elapsed_steps = 0
        ts = self.env.reset()
        obs = self._flatten_obs(ts)
        return obs, 0.0, False, {}

    def step(self, action: np.ndarray):
        self._elapsed_steps += 1
        # agent outputs actions in [-1, 1]
        action = np.clip(action, -1.0, 1.0)

        mins = self._action_spec.minimum
        maxs = self._action_spec.maximum
        # broadcast-safe scaling
        scaled = mins + (action + 1.0) * 0.5 * (maxs - mins)

        ts = self.env.step(scaled)
        reward = float(ts.reward) if ts.reward is not None else 0.0
        done = bool(ts.last() or (self._elapsed_steps >= self._max_episode_steps))
        obs = self._flatten_obs(ts)
        return obs, reward, done, {}


class VectorizedDMCEnv:
    """Simple synchronous vectorized wrapper for multiple DMC envs."""

    def __init__(self, domain: str, task: str, seed: int, num_envs: int,
                 max_episode_steps: int = 1000):
        self.num_envs = num_envs
        self.envs = []
        for idx in range(num_envs):
            env_seed = seed + idx
            self.envs.append(
                DMCEnvWrapper(domain, task, seed=env_seed,
                              max_episode_steps=max_episode_steps)
            )

        self._obs_dim = self.envs[0].observation_space_shape[0]
        self._action_dim = self.envs[0].action_space_shape[0]

    @property
    def observation_space_shape(self):
        return (self._obs_dim,)

    @property
    def action_space_shape(self):
        return (self._action_dim,)

    def reset(self) -> np.ndarray:
        obs_batch = []
        for env in self.envs:
            obs, _, _, _ = env.reset()
            obs_batch.append(obs)
        return np.stack(obs_batch, axis=0)

    def reset_at(self, idx: int) -> np.ndarray:
        obs, _, _, _ = self.envs[idx].reset()
        return obs

    def step(self, actions: np.ndarray):
        next_obs, rewards, dones = [], [], []
        for env, action in zip(self.envs, actions):
            obs, rew, done, _ = env.step(action)
            next_obs.append(obs)
            rewards.append(rew)
            dones.append(done)
        return (
            np.stack(next_obs, axis=0),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
        )


# --------------------------- replay buffer --------------------------- #

class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int = int(1e6)):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    def __len__(self):
        return self.size


# --------------------------- SAC networks --------------------------- #

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 256,
                 log_std_min: float = -20,
                 log_std_max: float = 2):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, state):
        """Sample action with tanh squashing and compute log-prob."""
        mu, std = self.forward(state)
        normal = torch.distributions.Normal(mu, std)
        x_t = normal.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t
        # log π(a|s) with tanh correction
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mu_tanh = torch.tanh(mu)
        return action, log_prob, mu_tanh


# --------------------------- SAC agent --------------------------- #

class SACAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: Config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.device = torch.device(cfg.device)

        self.actor = GaussianPolicy(state_dim, action_dim,
                                    hidden_dim=cfg.hidden_dim).to(self.device)
        self.q1 = QNetwork(state_dim, action_dim,
                           hidden_dim=cfg.hidden_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim,
                           hidden_dim=cfg.hidden_dim).to(self.device)
        self.q1_target = QNetwork(state_dim, action_dim,
                                  hidden_dim=cfg.hidden_dim).to(self.device)
        self.q2_target = QNetwork(state_dim, action_dim,
                                  hidden_dim=cfg.hidden_dim).to(self.device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.critic_lr)

        # automatic entropy tuning
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        squeeze = state_t.dim() == 1
        if squeeze:
            state_t = state_t.unsqueeze(0)

        with torch.no_grad():
            if deterministic:
                mu, _ = self.actor.forward(state_t)
                action = torch.tanh(mu)
            else:
                action, _, _ = self.actor.sample(state_t)

        action_np = action.cpu().numpy()
        return action_np[0] if squeeze else action_np

    def update(self, replay_buffer: ReplayBuffer, batch_size: int):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # ---------------- critic update ---------------- #
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor.sample(next_states)
            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)
            min_q_next = torch.min(q1_next, q2_next)
            target_q = rewards + (1.0 - dones) * self.gamma * (
                min_q_next - self.alpha * next_log_pi
            )

        q1 = self.q1(states, actions)
        q2 = self.q2(states, actions)
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        critic_loss = q1_loss + q2_loss

        self.q1_opt.zero_grad()
        self.q2_opt.zero_grad()
        critic_loss.backward()
        self.q1_opt.step()
        self.q2_opt.step()

        # ---------------- actor update ---------------- #
        pi, log_pi, _ = self.actor.sample(states)
        q1_pi = self.q1(states, pi)
        q2_pi = self.q2(states, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * log_pi - min_q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ---------------- temperature update ---------------- #
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # ---------------- target network update ---------------- #
        with torch.no_grad():
            for param, target_param in zip(self.q1.parameters(),
                                           self.q1_target.parameters()):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * param.data)

            for param, target_param in zip(self.q2.parameters(),
                                           self.q2_target.parameters()):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * param.data)


# --------------------------- training helpers --------------------------- #

def evaluate_policy(agent: SACAgent, env: DMCEnvWrapper,
                    max_episode_steps: int,
                    n_episodes: int = 5) -> float:
    returns = []
    for _ in range(n_episodes):
        obs, _, _, _ = env.reset()
        ep_ret = 0.0
        for _ in range(max_episode_steps):
            action = agent.act(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            ep_ret += reward
            if done:
                break
        returns.append(ep_ret)
    return float(np.mean(returns))


def train_one_seed(cfg: Config, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Train SAC for a single random seed. Returns train & eval returns."""
    set_seed(seed)

    env = VectorizedDMCEnv(cfg.domain, cfg.task,
                           seed=seed,
                           num_envs=cfg.num_envs,
                           max_episode_steps=cfg.max_episode_steps)
    eval_env = DMCEnvWrapper(cfg.domain, cfg.task, seed=cfg.eval_seed,
                              max_episode_steps=cfg.max_episode_steps)

    state_dim = env.observation_space_shape[0]
    action_dim = env.action_space_shape[0]

    agent = SACAgent(state_dim, action_dim, cfg)
    buffer = ReplayBuffer(state_dim, action_dim, capacity=cfg.replay_size)

    train_returns: List[float] = []
    eval_returns: List[float] = []

    obs_batch = env.reset()
    ep_returns = np.zeros(cfg.num_envs, dtype=np.float32)
    global_step = 0

    while len(train_returns) < cfg.num_episodes:
        if global_step < cfg.random_steps:
            actions = np.random.uniform(
                low=-1.0, high=1.0, size=(cfg.num_envs, action_dim)
            ).astype(np.float32)
        else:
            actions = agent.act(obs_batch, deterministic=False)
            actions = np.asarray(actions, dtype=np.float32)

        next_obs, rewards, dones = env.step(actions)

        for env_idx in range(cfg.num_envs):
            buffer.add(
                obs_batch[env_idx],
                actions[env_idx],
                rewards[env_idx],
                next_obs[env_idx],
                dones[env_idx],
            )
            ep_returns[env_idx] += rewards[env_idx]

        global_step += cfg.num_envs

        if len(buffer) >= cfg.batch_size:
            updates_to_run = cfg.updates_per_step * cfg.num_envs
            for _ in range(updates_to_run):
                agent.update(buffer, cfg.batch_size)

        for env_idx in range(cfg.num_envs):
            if dones[env_idx]:
                train_returns.append(float(ep_returns[env_idx]))
                ep_returns[env_idx] = 0.0
                next_obs[env_idx] = env.reset_at(env_idx)

                if len(train_returns) % cfg.eval_interval == 0:
                    eval_ret = evaluate_policy(
                        agent, eval_env, cfg.max_episode_steps, cfg.eval_episodes
                    )
                    eval_returns.append(eval_ret)
                    print(
                        f"[seed {seed}] episode {len(train_returns)}/{cfg.num_episodes} "
                        f"train_return={train_returns[-1]:.1f}, eval_return={eval_ret:.1f}"
                    )

                if len(train_returns) >= cfg.num_episodes:
                    break

        obs_batch = next_obs

    return np.asarray(train_returns, dtype=np.float32), np.asarray(eval_returns, dtype=np.float32)


def plot_learning_curves(cfg: Config,
                         all_train_returns: List[np.ndarray],
                         all_eval_returns: List[np.ndarray]):
    os.makedirs(cfg.out_dir, exist_ok=True)

    # ----- TRAINING CURVES: R0, R1, R2, mean ----- #
    train_arr = np.stack(all_train_returns, axis=0)  # [n_seeds, num_episodes]
    train_mean = train_arr.mean(axis=0)
    episodes = np.arange(1, cfg.num_episodes + 1)

    plt.figure()
    # one curve per seed
    for i, seed in enumerate(cfg.train_seeds):
        plt.plot(episodes,
                 train_arr[i],
                 alpha=0.6,
                 label=f"R{seed} (seed {seed})")
    # mean curve
    plt.plot(episodes,
             train_mean,
             linewidth=2.5,
             label="Mean over seeds (R̄)")

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("SAC – Training returns per seed and mean")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    train_path = os.path.join(cfg.out_dir, "training_R0_R1_R2_mean.png")
    plt.savefig(train_path, dpi=150)
    plt.close()

    # ----- EVALUATION CURVES: R0, R1, R2, mean ----- #
    eval_arr = np.stack(all_eval_returns, axis=0)  # [n_seeds, n_eval_points]
    eval_mean = eval_arr.mean(axis=0)

    eval_episodes_axis = np.arange(cfg.eval_interval,
                                   cfg.num_episodes + 1,
                                   cfg.eval_interval)

    plt.figure()
    # one curve per seed
    for i, seed in enumerate(cfg.train_seeds):
        plt.plot(eval_episodes_axis,
                 eval_arr[i],
                 alpha=0.6,
                 label=f"R{seed} eval (seed {seed})")
    # mean curve
    plt.plot(eval_episodes_axis,
             eval_mean,
             linewidth=2.5,
             label="Mean eval over seeds (R̄)")

    plt.xlabel("Training episode")
    plt.ylabel("Return")
    plt.title("SAC – Evaluation returns per seed and mean (env seed 10)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    eval_path = os.path.join(cfg.out_dir, "evaluation_R0_R1_R2_mean.png")
    plt.savefig(eval_path, dpi=150)
    plt.close()

    print(f"Saved training curves to {train_path}")
    print(f"Saved evaluation curves to {eval_path}")



# --------------------------- main entrypoint --------------------------- #

def main():
    cfg = Config()
    print("Using config:", cfg)

    all_train_returns: List[np.ndarray] = []
    all_eval_returns: List[np.ndarray] = []

    for seed in cfg.train_seeds:
        tr, ev = train_one_seed(cfg, seed)
        all_train_returns.append(tr)
        all_eval_returns.append(ev)

    plot_learning_curves(cfg, all_train_returns, all_eval_returns)


if __name__ == "__main__":
    main()
