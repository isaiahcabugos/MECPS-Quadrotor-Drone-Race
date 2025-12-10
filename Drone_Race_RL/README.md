# RL Drone Racing Training

Reinforcement learning for drone racing using TF-Agents and PPO.

## Structure
```
RL_Drone_Race/
├── src/
│   ├── simulator.py          # Clover drone simulator
│   ├── tf_agents_wrapper.py  # TF-Agents environment wrapper
│   ├── agent.py              # PPO agent creation
│   └── trainer.py            # Training loop
├── configs/
│   └── training_config.yaml  # Hyperparameters
├── checkpoints/              # Saved model checkpoints
├── logs/                     # TensorBoard logs
├── notebooks/                # Jupyter notebooks for analysis
├── train.py                  # Main training script
├── Dockerfile                # Docker container definition
└── README.md
```

## Local Training
```bash
python train.py
```

## Docker Training
```bash
# Build
docker build -t drone-racing-rl:latest .

# Run
docker run --gpus all \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/logs:/workspace/logs \
  drone-racing-rl:latest
```

## AWS Deployment

See `docs/aws_deployment.md`