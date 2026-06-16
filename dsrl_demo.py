"""Helper functions for the DSRL hopper demo notebook.

Everything here is meant to run *inside* the `dsrl:latest` container, where
MuJoCo / d4rl / dppo / stable-baselines3 are installed. The notebook stays thin
and calls into these helpers so the heavy logic is easy to test and reuse.
"""
import os
import sys
import math
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
os.environ.setdefault("MUJOCO_GL", "egl")  # GPU-accelerated headless rendering

BASE = "/workspace/dsrl"
for _p in (BASE, os.path.join(BASE, "dppo")):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np
import torch
from omegaconf import OmegaConf

import gym
import d4rl  # noqa: F401  (registers hopper-medium-v2)
import d4rl.gym_mujoco  # noqa: F401

from stable_baselines3 import DSRL
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from env_utils import ObservationWrapperGym, ActionChunkWrapper
from utils import load_base_policy, collect_rollouts


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(overrides=None):
    """Load the project's hopper config without Hydra and apply overrides."""
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
    OmegaConf.register_new_resolver("round_down", math.floor, replace=True)
    OmegaConf.register_new_resolver("now", lambda *a: "demo", replace=True)

    cfg = OmegaConf.load(os.path.join(BASE, "cfg/gym/dsrl_hopper.yaml"))
    cfg.use_wandb = False
    if overrides:
        cfg.merge_with(OmegaConf.create(overrides))
    OmegaConf.resolve(cfg)

    # Make the repo-relative ("./dppo/...") paths absolute so the demo works
    # regardless of the current working directory.
    def _abs(p):
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(BASE, p))
    cfg.normalization_path = _abs(cfg.normalization_path)
    cfg.base_policy_path = _abs(cfg.base_policy_path)
    cfg.model.network_path = _abs(cfg.model.network_path)
    cfg.dppo_path = _abs(cfg.dppo_path)
    return cfg


# --------------------------------------------------------------------------- #
# Environments
# --------------------------------------------------------------------------- #
def _make_chunk_env(cfg):
    """A single training env: normalized obs + action-chunk wrapper (matches train_dsrl)."""
    env = gym.make(cfg.env_name)
    env = ObservationWrapperGym(env, cfg.normalization_path)
    env = ActionChunkWrapper(env, cfg, max_episode_steps=cfg.env.max_episode_steps)
    return env


def make_train_env(cfg):
    # DummyVecEnv (single process) keeps the demo robust inside a Jupyter kernel —
    # SubprocVecEnv re-imports __main__ in workers, which breaks in notebooks.
    return make_vec_env(lambda: _make_chunk_env(cfg), n_envs=cfg.env.n_envs,
                        vec_env_cls=DummyVecEnv)


def make_eval_env(cfg, n_envs=1):
    return make_vec_env(lambda: _make_chunk_env(cfg), n_envs=n_envs,
                        vec_env_cls=DummyVecEnv)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def make_model(cfg, env):
    """Build the DSRL (DSRL-NA) algorithm exactly as train_dsrl.py does."""
    post_linear_modules = [torch.nn.LayerNorm] if cfg.train.use_layer_norm else None
    net_arch = [cfg.train.layer_size] * cfg.train.num_layers
    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, qf=net_arch),
        activation_fn=torch.nn.Tanh,
        log_std_init=0.0,
        post_linear_modules=post_linear_modules,
        n_critics=cfg.train.n_critics,
    )
    base_policy = load_base_policy(cfg)
    model = DSRL(
        "MlpPolicy", env,
        learning_rate=cfg.train.actor_lr,
        buffer_size=10_000_000,
        learning_starts=1,
        batch_size=cfg.train.batch_size,
        tau=cfg.train.tau,
        gamma=cfg.train.discount,
        train_freq=cfg.train.train_freq,
        gradient_steps=cfg.train.utd,
        ent_coef="auto" if cfg.train.ent_coef == -1 else cfg.train.ent_coef,
        target_update_interval=1,
        target_entropy="auto" if cfg.train.target_ent == -1 else cfg.train.target_ent,
        verbose=0,
        policy_kwargs=policy_kwargs,
        diffusion_policy=base_policy,
        diffusion_act_dim=(cfg.act_steps, cfg.action_dim),
        noise_critic_grad_steps=cfg.train.noise_critic_grad_steps,
        critic_backup_combine_type=cfg.train.critic_backup_combine_type,
    )
    return model, base_policy


def warmup_buffer(model, cfg, steps=200):
    """Seed the replay buffer with rollouts of the (un-steered) base diffusion policy."""
    if steps <= 0:
        return
    env = make_train_env(cfg)
    collect_rollouts(model, env, steps, model.diffusion_policy, cfg)
    env.close()


# --------------------------------------------------------------------------- #
# Rollout + rendering
# --------------------------------------------------------------------------- #
def _render_rollout(cfg, action_fn, max_chunks=250, seed=0):
    """Step a single rendered hopper env, choosing each action chunk via action_fn(obs)."""
    base = gym.make(cfg.env_name)
    base.seed(seed)
    env = ObservationWrapperGym(base, cfg.normalization_path)
    obs = env.reset()
    frames, total = [], 0.0
    for _ in range(max_chunks):
        chunk = action_fn(obs).reshape(cfg.act_steps, cfg.action_dim)
        done = False
        for i in range(cfg.act_steps):
            obs, r, done, _ = env.step(chunk[i])
            total += float(r)
            frames.append(base.render(mode="rgb_array"))
            if done:
                break
        if done:
            break
    base.close()
    return frames, total


def rollout_base_policy(base_policy, cfg, max_chunks=250, seed=0):
    """Behavior of the pretrained diffusion policy with standard Gaussian noise (the 'before')."""
    dev = cfg.model.device

    def action_fn(obs):
        noise = torch.randn(1, cfg.act_steps, cfg.action_dim, device=dev)
        a = base_policy(torch.tensor(obs[None], device=dev, dtype=torch.float32), noise)
        return np.asarray(a)

    return _render_rollout(cfg, action_fn, max_chunks, seed)


def rollout_dsrl(model, cfg, max_chunks=250, seed=0, deterministic=True):
    """Behavior of the DSRL-fine-tuned (steered-noise) policy (the 'after')."""
    def action_fn(obs):
        a, _ = model.predict_diffused(obs[None].astype(np.float32),
                                      deterministic=deterministic)
        return np.asarray(a)

    return _render_rollout(cfg, action_fn, max_chunks, seed)


def save_video(frames, path, fps=30):
    import imageio
    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=fps, macro_block_size=1)
    return path


def save_gif(frames, path, fps=15, width=320, every=2):
    """Save a downscaled GIF (renders inline in any notebook viewer, incl. VSCode)."""
    import imageio
    from PIL import Image
    sel = frames[::max(1, every)]
    out = []
    for f in sel:
        im = Image.fromarray(np.asarray(f))
        h = int(im.height * width / im.width)
        out.append(np.asarray(im.resize((width, h), Image.BILINEAR)))
    imageio.mimsave(path, out, fps=fps, loop=0)
    return path


def show_clip(frames, basename, fps=30, caption=None, width=360):
    """Save an mp4 (full res) + a GIF, and DISPLAY the GIF inline as an HTML <img>
    with a base64 data URI. This representation renders everywhere — JupyterLab,
    VSCode, and (crucially) the self-contained HTML export used for talks."""
    import base64
    from IPython.display import HTML, display
    save_video(frames, f"{basename}.mp4", fps=fps)
    gif = save_gif(frames, f"{basename}.gif")
    if caption:
        print(caption)
    b64 = base64.b64encode(open(gif, "rb").read()).decode("ascii")
    display(HTML(f'<img src="data:image/gif;base64,{b64}" width="{width}"/>'))


# --------------------------------------------------------------------------- #
# Multi-episode evaluation (no rendering — fast, for measuring performance)
# --------------------------------------------------------------------------- #
def _eval_returns(cfg, action_fn, n_episodes=50, max_chunks=300, seed=1000):
    env = make_eval_env(cfg, n_envs=1)
    env.seed(seed)
    rets, lens = [], []
    for _ in range(n_episodes):
        obs = env.reset()
        R, L, done = 0.0, 0, False
        for _ in range(max_chunks):
            obs, r, dones, _ = env.step(action_fn(obs))
            R += float(r[0]); L += cfg.act_steps
            if dones[0]:
                break
        rets.append(R); lens.append(L)
    env.close()
    return np.array(rets), np.array(lens)


def eval_returns_base(base_policy, cfg, n_episodes=50, max_chunks=300, seed=1000):
    """Returns of the pretrained diffusion policy with standard Gaussian noise."""
    dev = cfg.model.device

    def action_fn(obs):
        noise = torch.randn(obs.shape[0], cfg.act_steps, cfg.action_dim, device=dev)
        a = base_policy(torch.tensor(obs, device=dev, dtype=torch.float32), noise)
        return np.asarray(a).reshape(obs.shape[0], cfg.act_steps * cfg.action_dim)

    return _eval_returns(cfg, action_fn, n_episodes, max_chunks, seed)


def eval_returns_dsrl(model, cfg, n_episodes=50, max_chunks=300, seed=1000, deterministic=True):
    """Returns of the DSRL-fine-tuned (steered-noise) policy."""
    def action_fn(obs):
        a, _ = model.predict_diffused(obs.astype(np.float32), deterministic=deterministic)
        return np.asarray(a)

    return _eval_returns(cfg, action_fn, n_episodes, max_chunks, seed)


def summarize(name, rets, lens):
    m, sd = float(rets.mean()), float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    sem = sd / np.sqrt(len(rets)) if len(rets) else 0.0
    return dict(name=name, n=len(rets), mean=m, std=sd, sem=sem,
                ci95=1.96 * sem, median=float(np.median(rets)),
                min=float(rets.min()), max=float(rets.max()), mean_len=float(lens.mean()))


# --------------------------------------------------------------------------- #
# Training-metrics callback
# --------------------------------------------------------------------------- #
class DemoCallback(BaseCallback):
    """Collects training losses and periodic deterministic-eval returns for plotting."""

    def __init__(self, cfg, eval_every=2000, n_eval_episodes=5, max_chunks=250):
        super().__init__()
        self.cfg = cfg
        self.eval_every = eval_every
        self.n_eval_episodes = n_eval_episodes
        self.max_chunks = max_chunks
        self.act_chunk = cfg.act_steps
        self.n_envs = cfg.env.n_envs
        # logs
        self.t_loss, self.actor_loss, self.critic_loss, self.noise_critic_loss = [], [], [], []
        self.ent_coef = []
        self.t_eval, self.eval_return = [], []
        self.train_t, self.train_return = [], []
        self._eval_env = None
        self._next_eval = 0
        self._env_steps = 0

    def _env_timesteps(self):
        # number of underlying (per-action) env steps seen so far
        return self._env_steps

    def _on_step(self):
        self._env_steps += self.act_chunk * self.n_envs
        for info in self.locals["infos"]:
            if "episode" in info:
                self.train_t.append(self._env_steps)
                self.train_return.append(float(info["episode"]["r"]))
        # record losses from the SB3 logger
        v = self.model.logger.name_to_value
        if "train/actor_loss" in v:
            self.t_loss.append(self._env_steps)
            self.actor_loss.append(v.get("train/actor_loss", np.nan))
            self.critic_loss.append(v.get("train/critic_loss", np.nan))
            self.noise_critic_loss.append(v.get("train/noise_critic_loss", np.nan))
            self.ent_coef.append(v.get("train/ent_coef", np.nan))
        # periodic deterministic eval
        if self._env_steps >= self._next_eval:
            self._next_eval = self._env_steps + self.eval_every
            self.eval_return.append(self._evaluate())
            self.t_eval.append(self._env_steps)
            print(f"[eval] env_steps={self._env_steps:>7d}  mean_return={self.eval_return[-1]:8.1f}")
        return True

    def _evaluate(self):
        if self._eval_env is None:
            self._eval_env = make_eval_env(self.cfg, n_envs=1)
        env = self._eval_env
        returns = []
        for _ in range(self.n_eval_episodes):
            obs = env.reset()
            ep_ret, done = 0.0, False
            for _ in range(self.max_chunks):
                action, _ = self.model.predict_diffused(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                ep_ret += float(reward[0])
                if done[0]:
                    break
            returns.append(ep_ret)
        return float(np.mean(returns))
