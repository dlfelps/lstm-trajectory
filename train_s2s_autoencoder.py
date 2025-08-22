import pandas as pd
from pathlib import Path
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from itertools import pairwise

def load_data(file_path):
  def is_weekend(day):
    # 0 sunday, 1 monday, ..., 6 saturday
    temp = day % 7
    return int(temp == 0 or temp == 6)

  df = pd.read_csv(file_path)
  df = df[df['uid'] < 4000]
  data = []
  for _, g in df.groupby(['uid', 'd']):
    if len(g) < 2:
        continue
    temp = g.to_records(index=False).tolist()
    temp2 = list(map(lambda x: (x[0], x[3], x[4], is_weekend(x[1]), x[2]), temp))
    temp3 = pairwise(temp2)
    temp4 = list(map(lambda x: (x[0][0], x[0][1]/200, x[0][2]/200, x[0][3], x[0][4], x[1][4]), temp3))
    data.append(temp4)
  return data

class TrajectoryAutoencoderDataset(Dataset):
    """Dataset for trajectory autoencoder with data augmentation"""
    
    def __init__(self, trajectories, max_length=50, noise_std=0.0, augment_prob=1.0, training=True):
        """
        trajectories: List of trajectories, each trajectory is:
        [(user_id, lat, lon, is_weekend, timestamp), (user_id, lat, lon, is_weekend, timestamp), ...]
        For autoencoder, we use the full trajectory as both input and target
        """
        self.trajectories = trajectories
        self.max_length = max_length
        self.noise_std = noise_std
        self.augment_prob = augment_prob
        self.training = training    
   
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        
        # Truncate trajectory if too long
        if len(traj) > self.max_length:
            traj = traj[:self.max_length]
        
        # For autoencoder, use full trajectory as both input and target
        input_traj = traj
        target_traj = traj  # Same as input for reconstruction
        
        # Convert to tensors
        coordinates = []
        timestamps = []
        
        target_coordinates = []
        target_timestamps = []
        
        # Apply data augmentation if training and noise is enabled
        apply_augmentation = (self.training and 
                             self.noise_std > 0 and 
                             torch.rand(1).item() < self.augment_prob)
        
        # Process input trajectory
        for point in input_traj:
            user_id, x, y, isw, timestamp, next_timestamp = point
            
            # Add Gaussian noise to input coordinates if augmenting
            if apply_augmentation:
                noise_x = torch.normal(0, self.noise_std, size=(1,)).item()
                noise_y = torch.normal(0, self.noise_std, size=(1,)).item()
                x_noisy = x + noise_x
                y_noisy = y + noise_y
                coordinates.append([x_noisy, y_noisy])
            else:
                coordinates.append([x, y])
                
            timestamps.append([isw, timestamp, next_timestamp])
        
        # Process target trajectory (no noise for targets)
        for point in target_traj:
            _, x, y, isw, timestamp, next_timestamp = point
            target_coordinates.append([x, y])
            target_timestamps.append([isw, timestamp, next_timestamp])
        
        # Pad sequences
        input_seq_len = len(input_traj)
        
        while len(coordinates) < self.max_length:
            coordinates.append([1, 1])
            timestamps.append([2, 50, 50])
            target_coordinates.append([1, 1])
            target_timestamps.append([2, 50, 50])
        
        return {
            'coordinates': torch.tensor(coordinates, dtype=torch.float32),
            'timestamps': torch.tensor(timestamps, dtype=torch.float32),
            'target_coordinates': torch.tensor(target_coordinates, dtype=torch.float32),
            'target_timestamps': torch.tensor(target_timestamps, dtype=torch.float32),
            'seq_len': torch.tensor(input_seq_len, dtype=torch.long)
        }
    
    def set_training_mode(self, training):
        """Set training mode to enable/disable augmentation"""
        self.training = training

class SpatialEmbedding(nn.Module):
    """Embed lat/lon coordinates"""
    
    def __init__(self, input_dim=2, embedding_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, coordinates):
        x = F.relu(self.fc1(coordinates))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TemporalEmbedding(nn.Module):
    """Embed timestamps"""
    
    def __init__(self, input_dim=3, embedding_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, coordinates):
        x = F.relu(self.fc1(coordinates))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SequenceEncoder(nn.Module):
    """LSTM-based encoder for sequence-to-sequence autoencoder"""
    
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
    def forward(self, x, seq_lens):
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        
        # Forward pass through LSTM
        output, (hidden, cell) = self.lstm(x, (h0, c0))
        
        # Return final hidden state and all outputs
        return output, (hidden, cell)

class SequenceDecoder(nn.Module):
    """LSTM-based decoder for sequence-to-sequence autoencoder"""
    
    def __init__(self, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        self.lstm = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Project hidden state back to input dimension for autoregressive loop
        self.hidden_to_input = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, encoder_hidden, encoder_cell, target_seq_len, teacher_forcing=None, max_length=50):
        batch_size = encoder_hidden.size(1)
        max_seq_len = max_length
        
        # Initialize decoder input (zeros)
        decoder_input = torch.zeros(batch_size, 1, self.output_dim, device=encoder_hidden.device)
        
        # Initialize hidden state with encoder's final state
        hidden = encoder_hidden
        cell = encoder_cell
        
        outputs = []
        
        for t in range(max_seq_len):
            # Forward pass through decoder LSTM
            output, (hidden, cell) = self.lstm(decoder_input, (hidden, cell))
            outputs.append(output)
            
            # Use teacher forcing if provided, otherwise use own output
            if teacher_forcing is not None and t < teacher_forcing.size(1):
                decoder_input = teacher_forcing[:, t:t+1, :]
            else:
                # Project hidden state back to input dimension
                decoder_input = self.hidden_to_input(output)
        
        # Concatenate all outputs
        outputs = torch.cat(outputs, dim=1)
        return outputs

class TrajectoryAutoencoder(nn.Module):
    """Sequence-to-sequence autoencoder for trajectory reconstruction"""
    
    def __init__(self, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Embeddings
        self.spatial_embedding = SpatialEmbedding(input_dim=2, embedding_dim=embedding_dim)
        self.temporal_embedding = TemporalEmbedding(input_dim=3, embedding_dim=embedding_dim)
        
        # Combined embedding dimension
        combined_dim = embedding_dim * 2  # spatial + temporal
        
        # Encoder and Decoder
        self.encoder = SequenceEncoder(combined_dim, hidden_dim, num_layers, dropout)
        self.decoder = SequenceDecoder(hidden_dim, embedding_dim, num_layers, dropout)
        
        # Output projection layer (spatial only)
        self.spatial_projection = nn.Linear(hidden_dim, 2)  # Back to lat/lon
        
    def forward(self, coordinates, timestamps, seq_lens, teacher_forcing=False, target_coordinates=None, target_timestamps=None):
        batch_size, seq_len = coordinates.shape[0], coordinates.shape[1]
        
        # Create embeddings
        spatial_emb = self.spatial_embedding(coordinates)
        temporal_emb = self.temporal_embedding(timestamps)
        
        # Combine embeddings
        combined_emb = torch.cat([spatial_emb, temporal_emb], dim=-1)
        
        # Encode sequence
        encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder(combined_emb, seq_lens)
        
        # Prepare teacher forcing input if available (spatial only)
        teacher_forcing_input = None
        if teacher_forcing and target_coordinates is not None:
            teacher_forcing_input = self.spatial_embedding(target_coordinates)
        
        # Decode sequence (only spatial features)
        decoder_outputs = self.decoder(encoder_hidden, encoder_cell, seq_lens, teacher_forcing_input, max_length=50)
        
        # Project to spatial coordinates only
        reconstructed_coordinates = self.spatial_projection(decoder_outputs)
        
        return {
            'reconstructed_coordinates': reconstructed_coordinates,
            'encoder_outputs': encoder_outputs,
            'encoder_hidden': encoder_hidden,
            'latent_representation': encoder_hidden[-1]  # Last layer of encoder as latent
        }

class TrajectoryAutoencoderModel(nn.Module):
    """Main trajectory autoencoder model"""
    
    def __init__(self, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.autoencoder = TrajectoryAutoencoder(embedding_dim, hidden_dim, num_layers, dropout)
        
    def forward(self, batch, teacher_forcing=False):
        return self.autoencoder(
            batch['coordinates'], 
            batch['timestamps'],
            batch['seq_len'],
            teacher_forcing=teacher_forcing,
            target_coordinates=batch.get('target_coordinates'),
            target_timestamps=batch.get('target_timestamps')
        )
    
    def compute_loss(self, batch, outputs):
        """Compute reconstruction loss (spatial only)"""
        
        # Get targets and predictions
        target_coords = batch['target_coordinates']
        pred_coords = outputs['reconstructed_coordinates']
        
        # Create mask for valid positions
        seq_lens = batch['seq_len']
        batch_size, max_len = target_coords.shape[0], target_coords.shape[1]
        
        mask = torch.zeros(batch_size, max_len, device=target_coords.device)
        for i, length in enumerate(seq_lens):
            mask[i, :length] = 1
        
        # Expand mask for coordinates
        coord_mask = mask.unsqueeze(-1).expand_as(target_coords)
        
        # Compute masked coordinate loss
        coord_loss = F.mse_loss(pred_coords * coord_mask, target_coords * coord_mask, reduction='sum')
        coord_loss = coord_loss / coord_mask.sum()
        
        return {
            'total_loss': coord_loss,
            'coordinate_loss': coord_loss
        }

def train_autoencoder(learning_rate=0.001, batch_size=32, num_epochs=20, patience=10, 
                     warmup_epochs=3, gradient_clip=1.0, weight_decay=1e-5,
                     subset_fraction=0.25, max_batches_per_epoch=None,
                     noise_std=0.001, augment_prob=1.0, teacher_forcing_ratio=0.5):
    """Training loop for autoencoder"""
    
    # Load data
    print("Loading trajectory data...")
    trajectories = load_data("C:\\Users\\dlfel\\Projects\\python\\mobility\\data\\cityD-dataset.csv")
    
    full_dataset = TrajectoryAutoencoderDataset(trajectories, noise_std=noise_std, 
                                               augment_prob=augment_prob, training=True)
    
    print(f"Full dataset size: {len(full_dataset)} trajectories")
    if subset_fraction < 1.0:
        subset_size = int(len(full_dataset) * subset_fraction)
        print(f"Using {subset_fraction*100:.1f}% subset ({subset_size} trajectories) per epoch, reselecting each epoch")
    
    # Initialize model
    model = TrajectoryAutoencoderModel()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience//2
    )
    
    # Warmup scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    print(f"Teacher forcing ratio: {teacher_forcing_ratio}")
    
    # Early stopping variables
    best_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        # Select subset for this epoch
        if subset_fraction < 1.0:
            subset_size = int(len(full_dataset) * subset_fraction)
            indices = torch.randperm(len(full_dataset))[:subset_size]
            epoch_dataset = torch.utils.data.Subset(full_dataset, indices)
            dataloader = DataLoader(epoch_dataset, batch_size=batch_size, shuffle=True)
        else:
            dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
            
        total_loss = 0
        total_coord_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Stop early if max_batches_per_epoch is reached
            if max_batches_per_epoch is not None and batch_idx >= max_batches_per_epoch:
                print(f"Reached max batches per epoch ({max_batches_per_epoch}), stopping epoch early")
                break
                
            optimizer.zero_grad()
            
            # Randomly decide whether to use teacher forcing
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            
            # Forward pass
            outputs = model(batch, teacher_forcing=use_teacher_forcing)
            
            # Compute loss
            losses = model.compute_loss(batch, outputs)
            
            # Check for NaN losses
            if torch.isnan(losses['total_loss']):
                print(f"Warning: NaN loss detected at epoch {epoch}, batch {batch_idx}")
                continue
            
            # Backward pass
            losses['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            
            total_loss += losses['total_loss'].item()
            total_coord_loss += losses['coordinate_loss'].item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                      f"Total Loss: {losses['total_loss']:.4f}, "
                      f"Coord Loss: {losses['coordinate_loss']:.4f}, "
                      f"LR: {current_lr:.6f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        avg_coord_loss = total_coord_loss / num_batches if num_batches > 0 else float('inf')
        
        print(f"Epoch {epoch} completed:")
        print(f"  Average Total Loss: {avg_loss:.4f}")
        print(f"  Average Coordinate Loss: {avg_coord_loss:.4f}")
        
        # Learning rate scheduling
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step(avg_loss)
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            torch.save(best_checkpoint, 'best_autoencoder.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break
        
        # Save model checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }
        torch.save(checkpoint, f'autoencoder_epoch_{epoch}.pth')
        print(f"Model saved as autoencoder_epoch_{epoch}.pth")

def load_trained_autoencoder(checkpoint_path):
    """Load a trained autoencoder from checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model = TrajectoryAutoencoderModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint

def encode_trajectory(model, trajectory):
    """
    Encode trajectory to latent representation
    
    Args:
        model: trained TrajectoryAutoencoderModel
        trajectory: single trajectory to encode
    
    Returns:
        latent_representation: encoded trajectory representation
        encoder_outputs: full encoder outputs
    """
    model.eval()
    
    with torch.no_grad():
        # Convert trajectory to dataset format
        dataset = TrajectoryAutoencoderDataset([trajectory], training=False)
        batch = dataset[0]
        
        # Add batch dimension
        for key in batch:
            batch[key] = batch[key].unsqueeze(0)
        
        # Forward pass through model
        outputs = model(batch, teacher_forcing=False)
        
        return outputs['latent_representation'].squeeze(0), outputs['encoder_outputs'].squeeze(0)

def reconstruct_trajectory(model, trajectory):
    """
    Reconstruct trajectory using autoencoder
    
    Args:
        model: trained TrajectoryAutoencoderModel
        trajectory: trajectory to reconstruct
    
    Returns:
        reconstructed_coords: reconstructed coordinates
        reconstructed_times: reconstructed temporal features
        original_coords: original coordinates for comparison
        reconstruction_error: MSE between original and reconstructed
    """
    model.eval()
    
    with torch.no_grad():
        # Convert trajectory to dataset format
        dataset = TrajectoryAutoencoderDataset([trajectory], training=False)
        batch = dataset[0]
        
        # Store original data
        original_coords = batch['target_coordinates'].clone()
        seq_len = batch['seq_len'].item()
        
        # Add batch dimension
        for key in batch:
            batch[key] = batch[key].unsqueeze(0)
        
        # Forward pass
        outputs = model(batch, teacher_forcing=False)
        
        reconstructed_coords = outputs['reconstructed_coordinates'].squeeze(0)
        
        # Compute reconstruction error (only for valid sequence length)
        coord_error = F.mse_loss(reconstructed_coords[:seq_len], original_coords[:seq_len])
        
        return (reconstructed_coords, original_coords, coord_error.item())

def find_similar_trajectories_autoencoder(model, query_trajectory, trajectory_database, top_k=5):
    """
    Find similar trajectories using autoencoder latent representations
    
    Args:
        model: trained autoencoder model
        query_trajectory: trajectory to find similar ones for
        trajectory_database: list of trajectories to search through
        top_k: number of similar trajectories to return
    
    Returns:
        similar_trajectories: list of (trajectory_index, similarity_score) tuples
    """
    # Get latent representation for query trajectory
    query_latent, _ = encode_trajectory(model, query_trajectory)
    
    similarities = []
    
    for i, traj in enumerate(trajectory_database):
        # Get latent representation for database trajectory
        db_latent, _ = encode_trajectory(model, traj)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(
            query_latent.unsqueeze(0), 
            db_latent.unsqueeze(0), 
            dim=1
        ).item()
        
        similarities.append((i, similarity))
    
    # Sort by similarity and return top k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

if __name__ == "__main__":
    # Train the autoencoder
    train_autoencoder(
        learning_rate=0.001,
        batch_size=32,
        num_epochs=20,
        teacher_forcing_ratio=0.5,
        subset_fraction=0.25
    )