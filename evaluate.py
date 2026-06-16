"""Rigorous, multi-episode evaluation of DSRL vs the base diffusion policy on hopper.

Trains (or loads) the DSRL noise policy, then evaluates BOTH the base diffusion
policy (N(0,I) noise) and the DSRL-fine-tuned policy over many full episodes and
reports mean +/- 95% CI, so you can judge the *real* performance improvement.

Examples
--------
# quick correctness check (tiny train, few episodes):
python evaluate.py --timesteps 2000 --episodes 5

# convincing comparison (train ~longer, eval 100 episodes), save the policy:
python evaluate.py --timesteps 150000 --episodes 100 --save dsrl_hopper.pt

# re-evaluate a saved policy without retraining:
python evaluate.py --load dsrl_hopper.pt --episodes 100
"""
import argparse
import time

import numpy as np
import torch

import dsrl_demo as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=50000, help="RL fine-tuning timesteps")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--episodes", type=int, default=50, help="eval episodes per policy")
    ap.add_argument("--eval-chunks", type=int, default=300, help="max action-chunks per episode")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--save", default=None, help="save trained noise-policy weights (.pt)")
    ap.add_argument("--load", default=None, help="load noise-policy weights instead of training")
    args = ap.parse_args()

    cfg = D.load_config(overrides={"device": args.device})
    env = D.make_train_env(cfg)
    model, base_policy = D.make_model(cfg, env)

    if args.load:
        model.policy.load_state_dict(torch.load(args.load, map_location=args.device))
        print(f"Loaded noise policy from {args.load} (no training)")
    else:
        print(f"Fine-tuning DSRL for {args.timesteps} timesteps ...")
        D.warmup_buffer(model, cfg, steps=args.warmup)
        t0 = time.time()
        model.learn(total_timesteps=args.timesteps, progress_bar=False)
        print(f"  done in {time.time()-t0:.0f}s")
    if args.save:
        torch.save(model.policy.state_dict(), args.save)
        print(f"Saved noise policy to {args.save}")
    env.close()

    print(f"\nEvaluating over {args.episodes} episodes (seed {args.seed}) ...")
    b_ret, b_len = D.eval_returns_base(base_policy, cfg, args.episodes, args.eval_chunks, args.seed)
    d_ret, d_len = D.eval_returns_dsrl(model, cfg, args.episodes, args.eval_chunks, args.seed, True)
    s_ret, s_len = D.eval_returns_dsrl(model, cfg, args.episodes, args.eval_chunks, args.seed, False)

    rows = [D.summarize("base policy (N(0,I))", b_ret, b_len),
            D.summarize("DSRL (deterministic)", d_ret, d_len),
            D.summarize("DSRL (stochastic)", s_ret, s_len)]

    print("\n" + "=" * 78)
    print(f"{'policy':24s}{'mean return':>16s}{'95% CI':>12s}{'median':>10s}{'mean len':>12s}")
    print("-" * 78)
    for r in rows:
        print(f"{r['name']:24s}{r['mean']:>16.1f}{'+/-'+format(r['ci95'],'.0f'):>12s}"
              f"{r['median']:>10.1f}{r['mean_len']:>12.0f}")
    print("=" * 78)

    base_m = rows[0]["mean"]
    best = max(rows[1:], key=lambda r: r["mean"])
    delta = best["mean"] - base_m
    pct = 100.0 * delta / abs(base_m) if base_m else float("nan")
    # non-overlapping 95% CIs => clearly significant
    sig = (best["mean"] - best["ci95"]) > (base_m + rows[0]["ci95"])
    print(f"\nBest DSRL variant : {best['name']}")
    print(f"Improvement       : {delta:+.1f} return  ({pct:+.1f}% vs base)")
    print(f"95% CIs disjoint  : {sig}  (-> improvement is statistically clear)" if sig
          else f"95% CIs overlap   : train longer / more episodes for a clear signal")


if __name__ == "__main__":
    main()
