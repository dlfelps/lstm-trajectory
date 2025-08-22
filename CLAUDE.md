# Mobility Project - Claude Code Memory

## Project Overview
- Mobility trajectory analysis and machine learning project
- Uses PyTorch for neural network implementations
- Focuses on LSTM encoders and sequence-to-sequence autoencoders
- Dataset: cityD-dataset.csv with trajectory data

## Architecture & Models
- LSTM encoder for trajectory encoding
- Sequence-to-sequence autoencoder for trajectory reconstruction
- Spatial and temporal embedding layers
- Teacher forcing for training stability

## Data Format
- Trajectories: list of (user_id, lat, lon, is_weekend, timestamp, next_timestamp)
- Max sequence length: 50 (trajectories are padded/truncated)
- Coordinates normalized by dividing by 200
- User IDs capped at 4000

## Common Commands
- Training autoencoder: `uv run train_s2s_autoencoder.py`

## Development Notes
- All sequences are padded to max_length for batch processing
- Loss computation uses sequence masks to ignore padded positions
- Teacher forcing ratio of 0.5 recommended for autoencoder training
- Early stopping with patience=10 epochs

## File Structure
- `train_s2s_autoencoder.py`: Main autoencoder training script
- `data/cityD-dataset.csv`: Trajectory dataset
- Model checkpoints saved as `.pth` files

## Troubleshooting
- Tensor size mismatches often occur with sequence length handling
- Check teacher forcing bounds in decoder loops
- Verify sequence length consistency between inputs and targets