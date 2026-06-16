"""Generate demo/dsrl_hopper_demo.ipynb from source strings (valid ipynb v4)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Steering a Diffusion Policy with RL — DSRL demo (gym `hopper`)

This notebook demonstrates **DSRL** (*Diffusion Steering via Reinforcement Learning*,
Wagenmaker et al., CoRL 2025) on the lightest task the paper proposes — the MuJoCo
**`hopper-medium-v2`** locomotion task.

**The idea in one line:** instead of changing the weights of a pretrained diffusion
policy, DSRL learns a small RL agent that picks the *initial noise* fed into the
diffusion denoiser. Steering the noise steers the behavior — cheap and sample-efficient.

What this notebook shows, end-to-end, in one place:
1. Load the **pretrained diffusion policy** and build the **DSRL-NA** agent.
2. 🎥 Render the policy **before** fine-tuning (base policy + Gaussian noise).
3. 🚀 Run **RL fine-tuning in the noise space** while logging training metrics.
4. 📈 Plot the **training curves** (eval return + actor/critic/noise-critic losses).
5. 🎥 Render the policy **after** fine-tuning and compare.

> **Kernel:** run this inside the `dsrl:latest` container (e.g. `docker/notebook.sh`),
> which has MuJoCo + d4rl + dppo + stable-baselines3 installed and the GPU attached.""")

md(r"""## 0. Setup

`dsrl_demo.py` (next to this notebook) wraps the project's own building blocks
(`env_utils`, `utils`, the DPPO diffusion policy and the SB3 `DSRL` algorithm) so the
notebook stays readable. The training budget is parameterized — defaults give a clear
improvement on an RTX-4090 in ~15-20 min; lower `DEMO_TIMESTEPS` for a quicker look.""")

code(r"""import os, time
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Video, display
%matplotlib inline

import dsrl_demo as D

# ---- Demo budget (override via env vars without editing the notebook) ----
DEMO_TIMESTEPS = int(os.environ.get("DEMO_TIMESTEPS", 40000))  # SB3 timesteps of RL fine-tuning
WARMUP_STEPS   = int(os.environ.get("DEMO_WARMUP", 500))       # base-policy rollout steps to seed buffer
EVAL_EVERY     = int(os.environ.get("DEMO_EVAL_EVERY", 12000)) # eval cadence, in underlying env steps
N_EVAL         = int(os.environ.get("DEMO_N_EVAL", 3))         # eval episodes per checkpoint
EVAL_CHUNKS    = int(os.environ.get("DEMO_EVAL_CHUNKS", 250))  # max action-chunks per eval episode (250 = full)
VIDEO_CHUNKS   = int(os.environ.get("DEMO_VIDEO_CHUNKS", 250)) # max action-chunks per rendered episode

cfg = D.load_config(overrides={"device": "cuda:0"})
print(f"Task        : {cfg.env_name}")
print(f"Obs dim     : {cfg.obs_dim}   |  Action dim: {cfg.action_dim}  |  Action chunk: {cfg.act_steps}")
print(f"Algorithm   : {cfg.algorithm}  (DSRL-NA: learns a Q-function in the noise space)")
print(f"Device      : {cfg.device}")
print(f"Demo budget : {DEMO_TIMESTEPS} timesteps  (eval every {EVAL_EVERY} env-steps)")""")

md(r"""## 1. Load the pretrained diffusion policy & build the DSRL agent

The action space the RL agent sees is **not** the robot's torque space — it is the
`act_steps × action_dim` **noise space** of the diffusion policy. The agent proposes a
noise vector; the (frozen) diffusion policy denoises it into an action chunk that is
played in the environment.""")

code(r"""env = D.make_train_env(cfg)
model, base_policy = D.make_model(cfg, env)

print("Diffusion policy : DPPO DiffusionMLP, DDIM sampling, controllable noise (frozen)")
print("DSRL noise action space :", model.action_space.shape,
      f"= {cfg.act_steps} steps x {cfg.action_dim} dims")
n_params = sum(p.numel() for p in model.policy.parameters())
print(f"DSRL actor+critic params : {n_params:,}")""")

md(r"""## 2. 🎥 Behavior BEFORE fine-tuning

The starting point: the pretrained diffusion policy driven by **standard Gaussian
noise** `z ~ N(0, I)` — i.e. no steering. This is the "medium" behavior DSRL improves on.""")

code(r"""frames_before, ret_before = D.rollout_base_policy(base_policy, cfg, max_chunks=VIDEO_CHUNKS, seed=0)
# show_clip saves a full-res before.mp4 AND displays a GIF inline (renders in any viewer, incl. VSCode)
D.show_clip(frames_before, "before", fps=30,
            caption=f"Episode return (base policy, N(0,I) noise): {ret_before:.1f}  |  steps: {len(frames_before)}")""")

md(r"""## 3. 🚀 RL fine-tuning in the noise space (DSRL-NA)

We seed the replay buffer with a few base-policy rollouts, then run DSRL. The
`DemoCallback` records the losses every update and runs a short **deterministic eval**
every `EVAL_EVERY` env-steps so we can watch performance climb during fine-tuning.""")

code(r"""D.warmup_buffer(model, cfg, steps=WARMUP_STEPS)

cb = D.DemoCallback(cfg, eval_every=EVAL_EVERY, n_eval_episodes=N_EVAL, max_chunks=EVAL_CHUNKS)
t0 = time.time()
model.learn(total_timesteps=DEMO_TIMESTEPS, callback=cb, progress_bar=False)
print(f"\nFine-tuning done in {time.time()-t0:.0f}s")
print(f"Eval return:  start {cb.eval_return[0]:.0f}  ->  end {cb.eval_return[-1]:.0f}")""")

md(r"""## 4. 📈 Training curves

Left: deterministic **evaluation return** over the course of fine-tuning (the headline —
this is RL improving the policy by steering noise). Right: the SB3 **losses**, including
the DSRL-NA-specific **noise-critic loss** that distills the learned Q-function into the
noise space.""")

code(r"""fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))

ax[0].plot(cb.t_eval, cb.eval_return, "-o", color="tab:green")
ax[0].axhline(ret_before, ls="--", color="gray", label="base policy (N(0,I))")
ax[0].set_xlabel("environment steps"); ax[0].set_ylabel("eval return")
ax[0].set_title("RL fine-tuning progress (DSRL-NA)"); ax[0].legend(); ax[0].grid(alpha=0.3)

ax[1].plot(cb.t_loss, cb.actor_loss, label="actor_loss")
ax[1].plot(cb.t_loss, cb.critic_loss, label="critic_loss")
ax[1].plot(cb.t_loss, cb.noise_critic_loss, label="noise_critic_loss", alpha=0.8)
ax[1].set_xlabel("environment steps"); ax[1].set_ylabel("loss")
ax[1].set_title("Training losses"); ax[1].legend(); ax[1].grid(alpha=0.3)

plt.tight_layout(); plt.show()""")

md(r"""## 5. 🎥 Behavior AFTER fine-tuning

Same diffusion policy — but now the noise is chosen by the **learned DSRL actor**
(deterministic). Compare the gait and how long the hopper stays up against the "before"
clip.""")

code(r"""frames_after, ret_after = D.rollout_dsrl(model, cfg, max_chunks=VIDEO_CHUNKS, seed=0, deterministic=True)
D.show_clip(frames_after, "after", fps=30,
            caption=f"Episode return (DSRL-steered noise): {ret_after:.1f}  |  steps: {len(frames_after)}")""")

md(r"""## 6. Is it *really* better? — quantitative comparison

A single rollout is noisy. Here we evaluate the **pretrained** policy (Gaussian noise)
and the **DSRL** policy (deterministic + stochastic) over many full episodes and plot
the **mean return with 95% confidence intervals**. Non-overlapping bars ⇒ a real,
statistically clear improvement. This is the one slide to show an audience.""")

code(r"""N_BAR_EP = int(os.environ.get("DEMO_BAR_EPISODES", 20))
b_ret, b_len = D.eval_returns_base(base_policy, cfg, n_episodes=N_BAR_EP, max_chunks=EVAL_CHUNKS, seed=2000)
d_ret, d_len = D.eval_returns_dsrl(model, cfg, n_episodes=N_BAR_EP, max_chunks=EVAL_CHUNKS, seed=2000, deterministic=True)
s_ret, s_len = D.eval_returns_dsrl(model, cfg, n_episodes=N_BAR_EP, max_chunks=EVAL_CHUNKS, seed=2000, deterministic=False)

stats = [D.summarize("pretrained\n(base, N(0,I))", b_ret, b_len),
         D.summarize("DSRL\n(deterministic)", d_ret, d_len),
         D.summarize("DSRL\n(stochastic)", s_ret, s_len)]
labels = [s["name"] for s in stats]
means  = [s["mean"] for s in stats]
errs   = [s["ci95"] for s in stats]

fig, ax = plt.subplots(figsize=(6.2, 4.6))
bars = ax.bar(labels, means, yerr=errs, capsize=8,
              color=["gray", "tab:green", "tab:olive"], alpha=0.88)
for b, m in zip(bars, means):
    ax.text(b.get_x() + b.get_width()/2, m, f"{m:.0f}", ha="center", va="bottom", fontweight="bold")
ax.set_ylabel("episode return")
ax.set_title(f"Hopper return over {N_BAR_EP} episodes  (mean ± 95% CI)")
ax.grid(axis="y", alpha=0.3); plt.tight_layout(); plt.show()

base_m, base_ci = stats[0]["mean"], stats[0]["ci95"]
best = max(stats[1:], key=lambda s: s["mean"])
pct = 100*(best["mean"]-base_m)/abs(base_m)
sig = (best["mean"]-best["ci95"]) > (base_m + base_ci)
print(f"Best DSRL variant : {best['name'].replace(chr(10),' ')}")
print(f"Improvement       : {best['mean']-base_m:+.0f} return ({pct:+.1f}%) over {N_BAR_EP} episodes")
print(f"Statistically clear (95% CIs disjoint): {sig}")""")

md(r"""## 7. Summary""")

code(r"""print("="*54)
print(f"{'':22s}{'episode return':>16s}{'steps':>10s}")
print(f"{'base policy N(0,I)':22s}{ret_before:>16.1f}{len(frames_before):>10d}")
print(f"{'DSRL fine-tuned':22s}{ret_after:>16.1f}{len(frames_after):>10d}")
print("="*54)
print("\nDSRL fine-tuned the policy WITHOUT touching the diffusion weights —")
print("only by learning which initial noise to feed the (frozen) denoiser.")
env.close()""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9"},
}
with open("dsrl_hopper_demo.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote dsrl_hopper_demo.ipynb with", len(cells), "cells")
