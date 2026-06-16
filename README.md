# RL4IL-PoliMi-PhD-Course-AI-Applications-to-Industrial-Robotics
Material for the guest lectures on reinforcement learning for imitation learning in the PoliMi PhD Course on AI Applications to Industrial Robotics, organized by Prof. Loris Roveda

# DSRL Hands-On Session — Fine-Tuning a Diffusion Policy with Reinforcement Learning

**Steer a frozen diffusion policy with RL and run the entire pipeline live in one notebook.**

This repository accompanies the exercise session of the lecture:

**Robots that Learn from Imitation and Reinforcement:**
*Reinforcement Learning for Robot Policies in the Era of Imitation Learning*

The tutorial reproduces the main idea and code of the CoRL 2025 paper (please give credit and cite their work if you find it useful):

> **Steering Your Diffusion Policy with Latent Space Reinforcement Learning (DSRL)**
> Wagenmaker · Nakamoto · Zhang · Park · Yagoub · Nagabandi · Gupta · Levine

Instead of fine-tuning a diffusion policy directly, DSRL learns a reinforcement learning policy that selects the latent noise used by a **frozen diffusion policy**.

---

## Overview

The notebook demonstrates DSRL end-to-end on the Gymnasium **Hopper** benchmark:

1. Load a pretrained diffusion policy.
2. Render its behavior before fine-tuning.
3. Train a latent-noise RL policy.
4. Visualize learning curves during training.
5. Compare performance before and after adaptation.
6. Render the improved behavior.

The entire workflow is contained in a single notebook.

---

## Central Idea

Traditional policy adaptation updates neural network parameters.

DSRL instead learns a policy over the diffusion latent noise:

```text
State
  │
  ▼
Noise Policy πw
  │
  ▼
Latent Noise w
  │
  ▼
Frozen Diffusion Policy
  │
  ▼
Action
```

Only the noise-selection policy is trained.

The diffusion model remains completely frozen.

---

## Deliverables

### 1. Self-Contained HTML

`dsrl_hopper_demo.html`

A standalone export containing:

* Embedded videos
* Training plots
* Evaluation results

No Docker, Jupyter, GPU, or internet connection required.

Use this version for presentations and lectures.

---

### 2. Runnable Notebook

`dsrl_hopper_demo.ipynb`

The complete notebook used in the exercise session.

Run it inside the provided Docker container to reproduce all experiments.

---

### 3. Student Hands-On Environment

A self-contained Docker image:

```text
clmoro/dsrl-demo:latest
```

Includes:

* Notebook
* Dependencies
* Checkpoints
* Evaluation scripts

Students only need Docker and an NVIDIA GPU.

---

## Repository Structure

```text
.
├── dsrl_hopper_demo.ipynb
├── dsrl_hopper_demo.html
├── STUDENTS.md
├── dsrl_demo.py
├── evaluate.py
├── docker/
│   ├── Dockerfile.student
│   ├── build.sh
│   ├── notebook.sh
│   ├── export_demo.sh
│   ├── publish.sh
│   └── save_image.sh
└── checkpoints/
```

### Main Files

| File                     | Description                  |
| ------------------------ | ---------------------------- |
| `dsrl_hopper_demo.ipynb` | Main demonstration notebook  |
| `dsrl_hopper_demo.html`  | Offline presentation version |
| `dsrl_demo.py`           | DSRL utilities and helpers   |
| `evaluate.py`            | Rigorous evaluation script   |
| `STUDENTS.md`            | Student instructions         |

---

## Prerequisites

You need:

* NVIDIA GPU
* Docker
* NVIDIA Container Toolkit
* Approximately 18 GB free disk space

Verify Docker:

```bash
docker --version
```

Verify GPU access:

```bash
docker run --rm --gpus all \
nvidia/cuda:12.1.1-base-ubuntu22.04 \
nvidia-smi
```

Your GPU should appear in the output.

---

## Run the Demo

### Start the Container

```bash
docker run --rm --gpus all \
-p 8888:8888 \
clmoro/dsrl-demo:latest
```

### Open JupyterLab

Navigate to:

```text
http://127.0.0.1:8888/lab
```

Open:

```text
demo/dsrl_hopper_demo.ipynb
```

### Reproduce Results

Run:

```text
Run → Restart Kernel and Run All Cells
```

Videos and plots will be rendered inline.

### Different Port

If port 8888 is already in use:

```bash
docker run --rm --gpus all \
-p 8899:8888 \
clmoro/dsrl-demo:latest
```

Then open:

```text
http://127.0.0.1:8899/lab
```

---

## Notebook Workflow

### Step 1 — Load a Pretrained Policy

Load a diffusion policy trained through imitation learning.

### Step 2 — Evaluate Baseline Performance

Render the original policy and measure:

* Episode return
* Episode length
* Stability

### Step 3 — Train DSRL

Train a reinforcement learning policy in latent noise space while keeping the diffusion model frozen.

### Step 4 — Monitor Learning

Track:

* Evaluation return
* Actor loss
* Critic loss
* Noise critic loss

### Step 5 — Compare Policies

Evaluate:

* Original diffusion policy
* DSRL-enhanced policy

### Step 6 — Visualize Improvement

Render before-and-after videos and compare quantitative performance.

---

## Training Configuration

The notebook reads configuration from environment variables.

| Variable            | Description               | Default |
| ------------------- | ------------------------- | ------- |
| `DEMO_TIMESTEPS`    | RL fine-tuning timesteps  | `40000` |
| `DEMO_WARMUP`       | Buffer warmup steps       | `500`   |
| `DEMO_EVAL_EVERY`   | Evaluation interval       | `12000` |
| `DEMO_N_EVAL`       | Evaluation episodes       | `3`     |
| `DEMO_EVAL_CHUNKS`  | Maximum evaluation chunks | `250`   |
| `DEMO_VIDEO_CHUNKS` | Maximum rendering chunks  | `250`   |

For a shorter demonstration:

```bash
DEMO_TIMESTEPS=8000
```

---

## Evaluation

The notebook includes:

* Training curves
* Evaluation curves
* Statistical comparisons
* Confidence intervals

### Standalone Evaluation

Train and save:

```bash
python evaluate.py \
    --timesteps 150000 \
    --episodes 100 \
    --save dsrl_hopper.pt
```

Evaluate a saved policy:

```bash
python evaluate.py \
    --load dsrl_hopper.pt \
    --episodes 200
```

---

## Metrics to Monitor

### Episode Return

Primary performance metric.

Higher is better.

### Episode Length

For Hopper:

* Longer episodes generally indicate better locomotion.
* Early termination indicates failure.

### Evaluation Return

Used to monitor improvement during training.

### Confidence Intervals

95% confidence intervals are reported for rigorous comparison.

---

## What You Are Looking At

### Before Training

```text
z ~ N(0, I)
        ↓
Frozen Diffusion Policy
        ↓
Actions
```

The latent noise is sampled randomly.

### After DSRL

```text
RL Policy
     ↓
Latent Noise z
     ↓
Frozen Diffusion Policy
     ↓
Actions
```

The latent noise is chosen strategically to maximize return.

The diffusion model never changes.

---

## Key Takeaway

DSRL demonstrates a powerful paradigm for adapting foundation policies:

* Keep the diffusion model frozen.
* Train only a lightweight latent controller.
* Preserve demonstrated behavior.
* Improve performance with reinforcement learning.
* Avoid expensive diffusion-model fine-tuning.

This reflects a broader trend in modern robot learning:

> **Pretrain with imitation learning, then use reinforcement learning to surpass the demonstrations.**

---

## Reference

```bibtex
@article{wagenmaker2025dsrl,
  title={Steering Your Diffusion Policy with Latent Space Reinforcement Learning},
  author={Wagenmaker, Andrew and Nakamoto, Mitsuhiko and Zhang, Yunchu and Park, Suneel and Yagoub, Omar and Nagabandi, Anusha and Gupta, Abhinav and Levine, Sergey},
  year={2025}
}
```
