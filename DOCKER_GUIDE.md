# Docker Setup Guide for RL Drone Racing

Complete guide for training autonomous drone racing agents using Docker containers.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Dockerfile Overview](#dockerfile-overview)
- [Training Phases](#training-phases)
- [Common Workflows](#common-workflows)
- [Volume Mounts](#volume-mounts)
- [Troubleshooting](#troubleshooting)
- [AWS Deployment](#aws-deployment)

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install Docker
# Windows: Docker Desktop (with WSL2)
# Linux: docker, nvidia-docker2
# Mac: Docker Desktop

# Verify installation
docker --version
docker run hello-world

# For GPU training (Linux only)
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Project Structure

```
RL_Drone_Race/
├── Dockerfile                    # Main GPU training (Phase 1)
├── Dockerfile.cpu               # CPU-only training (Phase 1)
├── Dockerfile.cpufinetune       # Legacy CPU fine-tuning
├── Dockerfile.phase2            # Phase 2 residual fine-tuning
├── .dockerignore                # Docker ignore patterns
│
├── train.py                     # Main training script
├── finetune_with_residuals.py   # Phase 2 fine-tuning
├── test_train.py                # Quick training test (60k steps)
├── verify_residual_pipeline.py  # Pre-flight checks
├── collect_residuals.py         # Residual model creation
│
├── src/
│   ├── simulator.py             # Clover drone physics
│   ├── tf_agents_wrapper.py     # TF-Agents environment
│   ├── agent.py                 # PPO agent setup
│   └── trainer.py               # Training loop
│
├── checkpoints/                 # Saved models (mounted volume)
├── outputs/                     # Results, logs, plots (mounted volume)
└── dynamics_residual_model.pkl  # Pre-trained residual model
```

---

## 📦 Dockerfile Overview

### 1. `Dockerfile` - Main GPU Training (Phase 1)

**Purpose:** Train a new policy from scratch using GPU acceleration.

**Hardware:** NVIDIA GPU with CUDA 11.8+

**Use when:** Starting fresh, training specialist or generalist policies

```bash
# Build
docker build -f Dockerfile -t drone-racing-gpu:latest .

# Run (100M steps with 50 parallel envs)
docker run --gpus all \
  -v $(pwd)/checkpoints:/workspace/checkpoints:rw \
  -v $(pwd)/outputs:/workspace/outputs:rw \
  --name drone-train-gpu \
  drone-racing-gpu:latest \
  python3 train.py \
    --total-steps 100000000 \
    --num-envs 50 \
    --checkpoint-freq 1000000

# Monitor
docker logs -f drone-train-gpu
```

**Training time:** ~25 Hours with i5-13600k and no GPU support. Unsure how much a GPU will affect this.

---

### 2. `Dockerfile.cpu` - CPU-Only Training (Phase 1)

**Purpose:** Train without GPU (slower but more portable)

**Hardware:** Any CPU with 16+ GB RAM. In testing, RAM usage spiked at the beginning but lowered near the end. Unsure what caused that.

**Use when:** No GPU available, testing, small experiments

```bash
# Build
docker build -f Dockerfile.cpu -t drone-racing-cpu:latest .

# Run (smaller config)
docker run \
  -v $(pwd)/checkpoints:/workspace/checkpoints:rw \
  -v $(pwd)/outputs:/workspace/outputs:rw \
  --name drone-train-cpu \
  drone-racing-cpu:latest \
  python3 train.py \
    --total-steps 10000000 \
    --num-envs 20 \
    --checkpoint-freq 500000

# Monitor
docker logs -f drone-train-cpu
```

**Training time:** ~25 hours for 100M steps. Refer to report.

---

### 3. `Dockerfile.phase2` - Residual Fine-Tuning (Phase 2)

**Purpose:** Fine-tune trained policy with sim-to-real residual models

**Hardware:** CPU sufficient (50 envs uses ~16 GB RAM, all cores)

**Use when:** You have a Phase 1 checkpoint + residual model

```bash
# Build
docker build -f Dockerfile.phase2 -t drone-racing-phase2:latest .

# Run (continue from 100M → 120M with residuals)
docker run -d \
  -v $(pwd)/checkpoints:/workspace/checkpoints:rw \
  -v $(pwd)/outputs:/workspace/outputs:rw \
  -v $(pwd)/dynamics_residual_model.pkl:/workspace/dynamics_residual_model.pkl:ro \
  --name drone-finetune \
  drone-racing-phase2:latest \
  python3 finetune_with_residuals.py \
    --checkpoint-dir /workspace/checkpoints \
    --residual-model /workspace/dynamics_residual_model.pkl \
    --steps 20000000 \
    --num-envs 50

# Monitor
docker logs -f drone-finetune
```

**Training time:** ~5 hours for 20M steps (depends on CPU)

---

### 4. `Dockerfile.cpufinetune` - Legacy CPU Fine-Tuning

**Purpose:** Older version of Phase 2 fine-tuning

**Status:** Deprecated, use `Dockerfile.phase2` instead

---

## 🎯 Training Phases

### Phase 1: Simulation Training (0 → 100M steps)

**Objective:** Learn racing behavior in lightweight simulator

**Duration:** Refer to report/presentation for expected time. Did not test with GPU.

**Output:** `checkpoints/ckpt-100000000*`, `policy/policy_step_100000000/`

**Dockerfile:** `Dockerfile` (GPU) or `Dockerfile.cpu` (CPU)

**What it learns:**
- Gate navigation and sequencing
- Velocity control and trajectory optimization
- Camera pointing (perception reward)
- Action smoothness

---

### Phase 2: Sim-to-Real Transfer (100M → 120M steps)

**Objective:** Adapt policy to realistic dynamics and noisy perception

**Duration:** Took ~5 hours on specifications in report.

**Output:** `checkpoints/ckpt-120000000*`, `policy/policy_FINAL/`

**Dockerfile:** `Dockerfile.phase2`

**Prerequisites:**
1. Phase 1 checkpoint at 100M steps
2. Residual model (`dynamics_residual_model.pkl`)

**What it learns:**
- Compensate for dynamics mismatch (tau=0.1 → tau=1.0)

**Important Note:**
- There seems to be something incorrectly done on this step. The model does not improve as expected.

---

## 💼 Common Workflows

### Workflow 1: Train from Scratch (Phase 1 Only)

```bash
# Build image
docker build -f Dockerfile -t drone-racing-gpu:latest .

# Train for 100M steps
docker run --gpus all \
  -v $(pwd)/checkpoints:/workspace/checkpoints:rw \
  -v $(pwd)/outputs:/workspace/outputs:rw \
  --name drone-phase1 \
  drone-racing-gpu:latest \
  python3 train.py \
    --total-steps 100000000 \
    --num-envs 50 \
    --checkpoint-freq 1000000

# Wait to finish...

# Test trained policy
docker run --gpus all \
  -v $(pwd)/checkpoints:/workspace/checkpoints:ro \
  -v $(pwd)/outputs:/workspace/outputs:rw \
  --rm \
  drone-racing-gpu:latest \
  python3 visualizetests.py

# View results
ls checkpoints/policy/policy_step_100000000/
ls outputs/trajectory_debug.png
```

---

### Workflow 2: Full Pipeline (Phase 1 → Phase 2)

```bash
# Step 1: Train Phase 1 (see Workflow 1)
# ... wait for 100M steps completion ...

# Step 2: Verify checkpoint and residual model
docker build -f Dockerfile.phase2 -t drone-racing-phase2:latest .

docker run --rm \
  -v $(pwd)/checkpoints:/workspace/checkpoints:ro \
  -v $(pwd)/dynamics_residual_model.pkl:/workspace/dynamics_residual_model.pkl:ro \
  drone-racing-phase2:latest \
  python3 verify_residual_pipeline.py \
    --checkpoint-dir /workspace/checkpoints \
    --residual-model /workspace/dynamics_residual_model.pkl

# Should see: ✅ ALL TESTS PASSED!

# Step 3: Run Phase 2 fine-tuning
docker run -d \
  -v $(pwd)/checkpoints:/workspace/checkpoints:rw \
  -v $(pwd)/outputs:/workspace/outputs:rw \
  -v $(pwd)/dynamics_residual_model.pkl:/workspace/dynamics_residual_model.pkl:ro \
  --name drone-phase2 \
  drone-racing-phase2:latest \
  python3 finetune_with_residuals.py \
    --checkpoint-dir /workspace/checkpoints \
    --residual-model /workspace/dynamics_residual_model.pkl \
    --steps 20000000 \
    --num-envs 50

# Monitor progress
docker logs -f drone-phase2

# Wait ~to finish...

# Extract final policy
docker exec drone-phase2 ls /workspace/checkpoints/policy/
# Look for: policy_FINAL/ or policy_step_120000000/
```

---

### Workflow 3: Quick Test (60k steps)

**Purpose:** Verify setup works before committing to long training

```bash
# Build
docker build -f Dockerfile -t drone-racing-gpu:latest .

# Run quick test
docker run --gpus all \
  -v $(pwd)/outputs:/workspace/outputs:rw \
  --rm \
  drone-racing-gpu:latest \
  python3 test_train.py

# Should complete in ~2-5 minutes
# Check output: "✅ TEST PASSED - Training completed successfully!"
```

---

### Workflow 4: Resume Interrupted Training

**Important Note:**
- Number of environments must stay consistent between training sessions.

```bash
# Training was interrupted at step 75,000,000
# Checkpoint saved automatically

# Resume from last checkpoint (automatically detected)
docker run --gpus all \
  -v $(pwd)/checkpoints:/workspace/checkpoints:rw \
  -v $(pwd)/outputs:/workspace/outputs:rw \
  --name drone-resume \
  drone-racing-gpu:latest \
  python3 train.py \
    --total-steps 100000000 \
    --num-envs 50

# Will automatically load ckpt-75000000 and continue to 100M
```

---

### Workflow 5: Create Custom Residual Model

```bash
# Step 1: Collect data in Gazebo (requires ROS/Gazebo setup)
docker run --rm \
  -v $(pwd)/checkpoints:/workspace/checkpoints:ro \
  -v $(pwd)/outputs:/workspace/outputs:rw \
  --network host \
  drone-racing-phase2:latest \
  python3 collect_residuals.py collect \
    --policy /workspace/checkpoints/policy/policy_step_100000000 \
    --output /workspace/outputs/gazebo_data.pkl \
    --duration 60

# Step 2: Fit residual model
docker run --rm \
  -v $(pwd)/outputs:/workspace/outputs:rw \
  drone-racing-phase2:latest \
  python3 collect_residuals.py fit \
    --input /workspace/outputs/gazebo_data.pkl \
    --output /workspace/outputs/my_residual_model.pkl \
    --k 5

# Step 3: Use your custom model
docker run -d \
  -v $(pwd)/checkpoints:/workspace/checkpoints:rw \
  -v $(pwd)/outputs:/workspace/outputs:rw \
  --name drone-custom-residual \
  drone-racing-phase2:latest \
  python3 finetune_with_residuals.py \
    --checkpoint-dir /workspace/checkpoints \
    --residual-model /workspace/outputs/my_residual_model.pkl \
    --steps 20000000 \
    --num-envs 50
```

---

## 📂 Volume Mounts

### Required Mounts

```bash
-v $(pwd)/checkpoints:/workspace/checkpoints:rw    # Trained models
-v $(pwd)/outputs:/workspace/outputs:rw            # Logs, plots, results
```

### Optional Mounts

```bash
-v $(pwd)/dynamics_residual_model.pkl:/workspace/dynamics_residual_model.pkl:ro  # Phase 2 only
-v $(pwd)/configs:/workspace/configs:ro                                           # Custom configs
```

### Mount Permissions

- `:rw` - Read-write (default, use for checkpoints/outputs)
- `:ro` - Read-only (use for residual models, configs)

**Best practice:** Mount checkpoints as `:rw` so training can save progress

---

## 🔍 Monitoring Training

### View Logs (Real-time)

```bash
# Follow logs
docker logs -f drone-train-gpu

# Last 100 lines
docker logs --tail 100 drone-train-gpu

# Save logs to file
docker logs drone-train-gpu > training_log.txt 2>&1
```

### Check Progress

```bash
# Exec into container
docker exec -it drone-train-gpu bash

# Inside container:
ls -lh /workspace/checkpoints/
cat /workspace/checkpoints/checkpoint  # Shows latest checkpoint
tail -f /workspace/outputs/logs_finetuned/events.out.tfevents.*  # TensorBoard logs
```

### TensorBoard (if enabled)

```bash
# Expose TensorBoard port
docker run --gpus all \
  -v $(pwd)/checkpoints:/workspace/checkpoints:rw \
  -v $(pwd)/outputs:/workspace/outputs:rw \
  -p 6006:6006 \
  --name drone-train \
  drone-racing-gpu:latest \
  bash -c "tensorboard --logdir /workspace/outputs/logs_finetuned --host 0.0.0.0 & python3 train.py --total-steps 100000000"

# View at http://localhost:6006
```

---

## 🛠️ Troubleshooting

### Issue 1: Out of Memory (GPU)

**Symptoms:** `ResourceExhaustedError`, container crashes

**Solutions:**
```bash
# Reduce parallel environments
--num-envs 20  # Instead of 50

# Use smaller batch size (modify agent.py)
# Or train on CPU instead
docker build -f Dockerfile.cpu -t drone-racing-cpu:latest .
```

---

### Issue 2: Container Exits Immediately

**Check logs:**
```bash
docker logs drone-train-gpu
```

**Common causes:**
- Missing volume mounts → Check `-v` paths exist
- Wrong Python path → Should be `python3` not `python`
- Checkpoint directory permissions → Use `:rw` not `:ro`

---

### Issue 3: Permission Denied (Checkpoints)

**Symptoms:** `PermissionDeniedError: Read-only file system`

**Solution:**
```bash
# Change mount from :ro to :rw
-v $(pwd)/checkpoints:/workspace/checkpoints:rw

# Or use separate save directory
--save-checkpoint-dir /workspace/outputs/checkpoints_finetuned
```

---

### Issue 4: Slow Training (CPU)

**Symptoms:** <10,000 FPS, long iteration times

**Solutions:**
```bash
# Reduce parallel environments
--num-envs 10  # Fewer workers = less overhead

# Use more CPU cores
docker run --cpus 16  # Allocate specific cores

# Or switch to GPU training
docker build -f Dockerfile -t drone-racing-gpu:latest .
docker run --gpus all ...
```

---

### Issue 5: "Replay Buffer Shape Mismatch"

**Symptoms:** `ValueError: Shapes (3000, 3) and (75000, 3) are incompatible`

**Cause:** Checkpoint trained with 50 envs, trying to load with 20 envs

**Solution:**
```bash
# Match the number of environments
--num-envs 50  # Same as original training

# The verify script handles this automatically
python3 verify_residual_pipeline.py  # Skips replay buffer
```

---

### Issue 6: Multiprocessing Already Initialized

**Symptoms:** `ValueError: Multiprocessing already initialized`

**Fix:** Already handled in updated scripts (`try/except` wrapper)

If you see this, update your scripts:
```python
from tf_agents.system import system_multiprocessing
try:
    system_multiprocessing.enable_interactive_mode()
except ValueError:
    pass  # Already initialized
```

---

## ☁️ AWS Deployment

**Important Note:**
- The following section was added by the gen-ai used to assist in the creation of this document. AWS was not used by this project, but I have decided to keep this section in case it is useful for future project contributors.


### EC2 Instance Types

**Phase 1 (GPU Training):**
- `g4dn.xlarge` - 1x T4 GPU, $0.50/hr (budget)
- `g5.xlarge` - 1x A10G, $1.00/hr (balanced)
- `p3.2xlarge` - 1x V100, $3.00/hr (fast)

**Phase 2 (CPU Fine-tuning):**
- `c6i.8xlarge` - 32 vCPU, $1.36/hr (recommended)
- `c7i.12xlarge` - 48 vCPU, $2.04/hr (fastest)

### Example Launch Script

```bash
#!/bin/bash
# deploy_aws.sh

# Install Docker on EC2 Ubuntu instance
sudo apt-get update
sudo apt-get install -y docker.io nvidia-docker2
sudo systemctl restart docker

# Clone repo
git clone <your-repo> RL_Drone_Race
cd RL_Drone_Race

# Build image
docker build -f Dockerfile -t drone-racing-gpu:latest .

# Run training (detached)
docker run -d --gpus all \
  -v $(pwd)/checkpoints:/workspace/checkpoints:rw \
  -v $(pwd)/outputs:/workspace/outputs:rw \
  --name drone-train \
  drone-racing-gpu:latest \
  python3 train.py \
    --total-steps 100000000 \
    --num-envs 50 \
    --checkpoint-freq 1000000

# Monitor
docker logs -f drone-train

# Download results when done
aws s3 sync checkpoints/ s3://my-bucket/drone-racing/checkpoints/
aws s3 sync outputs/ s3://my-bucket/drone-racing/outputs/
```

---

## 📊 Expected Performance

### Phase 1 Training (100M steps)

| Hardware | Envs | FPS | Time | Cost (AWS) |
|----------|------|-----|------|------------|
| RTX 3090 | 50 | ~200K | 6 hrs | N/A |
| RTX 4090 | 50 | ~300K | 4 hrs | N/A |
| T4 (AWS) | 30 | ~80K | 15 hrs | $7.50 |
| V100 (AWS) | 50 | ~150K | 8 hrs | $24 |
| CPU 32-core | 20 | ~15K | 80 hrs | $108 |

### Phase 2 Fine-tuning (20M steps)

| Hardware | Envs | FPS | Time | Cost (AWS) |
|----------|------|-----|------|------------|
| CPU 16-core | 50 | ~180K | 2 hrs | N/A |
| CPU 32-core | 50 | ~250K | 1.5 hrs | $2 |
| CPU 48-core | 50 | ~300K | 1 hr | $2 |

---

## 🎓 Learning Resources

### Understanding the Code

- `src/simulator.py` - PX4-style cascaded control, Swift reward function
- `src/agent.py` - PPO hyperparameters (match Swift paper)
- `src/trainer.py` - Training loop, checkpointing, metrics
- `finetune_with_residuals.py` - Residual learning integration

### Key Papers

- **Swift (Nature 2023):** Champion-level drone racing using deep RL
- **PPO (Schulman 2017):** Proximal Policy Optimization algorithm

### Hyperparameters (Swift-aligned)

```python
# PPO Config (src/agent.py)
learning_rate = 3e-4
num_epochs = 10
discount = 0.99
gae_lambda = 0.95
clip_ratio = 0.2
entropy_coef = 0.01

# Reward Weights (src/simulator.py)
lambda1 = 1.0    # r_progress
lambda2 = 0.02   # r_perc (perception)
lambda3 = 10.0   # r_perc exp coefficient
lambda4 = 0.0001 # r_smooth (action)
lambda5 = 0.0002 # r_smooth (change)
```

---

## 🏁 Next Steps

After successful Docker training:

1. **Extract trained policy:**
   ```bash
   cp checkpoints/policy/policy_step_120000000/ ./deployed_policy/
   ```

2. **Deploy to hardware:**
   - Convert to TensorFlow Lite (Raspberry Pi)
   - Test in Gazebo simulation first
   - Real Clover hardware flight

3. **Iterate:**
   - Collect real-world data
   - Fit new residual models
   - Fine-tune with updated residuals

---
