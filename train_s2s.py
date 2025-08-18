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
    temp = g.to_records(index=False).tolist()
    temp2 = list(map(lambda x: (x[0], x[3], x[4], is_weekend(x[1]), x[2]), temp))
    temp3 = pairwise(temp2)
    temp4 = list(map(lambda x: (x[0][0], x[0][1], x[0][2], x[0][3], x[0][4], x[1][4]), temp3))
    data.append(temp4)
  return data

class TrajectoryDataset(Dataset):
    """Dataset for trajectory sequences with user IDs"""
    
    def __init__(self, trajectories, max_length=50):
        """
        trajectories: List of trajectories, each trajectory is:
        [(user_id, lat, lon, is_weekend, timestamp), (user_id, lat, lon, is_weekend, timestamp), ...]
        user_id should already be integer indices
        """
        self.trajectories = trajectories
        self.max_length = max_length
        self.num_users = 4000    
   
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        
        # Truncate or pad trajectory
        if len(traj) > self.max_length:
            traj = traj[:self.max_length]
        
        # Convert to tensors
        user_ids = []
        coordinates = []
        timestamps = []
        
        for point in traj:
            user_id, x, y, isw, timestamp, next_timestamp = point
            user_ids.append(user_id)  # user_id is already an integer index
            coordinates.append([x, y])            
            timestamps.append([isw,timestamp, next_timestamp])
        
        # Pad sequences
        seq_len = len(traj)
        pad_token = self.num_users #not an actual user id
        while len(user_ids) < self.max_length:
            user_ids.append(pad_token)  # padding token
            coordinates.append([201,201])
            timestamps.append([2,50, 50])
        
        return {
            'user_ids': torch.tensor(user_ids, dtype=torch.long),
            'coordinates': torch.tensor(coordinates, dtype=torch.float32),
            'timestamps': torch.tensor(timestamps, dtype=torch.float32),
            'seq_len': torch.tensor(seq_len, dtype=torch.long)
        }

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

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape
        
        # Linear projections
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # Add head and query dimensions
            scores.masked_fill_(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        output = self.out_proj(attn_output)
        
        return output, attn_weights.mean(dim=1)  # Average attention weights across heads

class SpatialTemporalAttention(nn.Module):
    """Cross-attention between spatial and temporal features"""
    
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.spatial_temporal_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.temporal_spatial_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, spatial_emb, temporal_emb, mask=None):
        # Spatial attending to temporal
        spatial_attended, _ = self.spatial_temporal_attn(
            spatial_emb + temporal_emb, mask=mask
        )
        spatial_emb = self.norm1(spatial_emb + self.dropout(spatial_attended))
        
        # Temporal attending to spatial
        temporal_attended, _ = self.temporal_spatial_attn(
            temporal_emb + spatial_emb, mask=mask
        )
        temporal_emb = self.norm2(temporal_emb + self.dropout(temporal_attended))
        
        return spatial_emb, temporal_emb

class TransformerBlock(nn.Module):
    """Transformer block with multi-head self-attention and feed-forward network"""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, attn_weights = self.self_attention(x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x, attn_weights

class HybridTransformerLSTMEncoder(nn.Module):
    """Hybrid Transformer-LSTM Encoder with spatial-temporal attention"""
    
    def __init__(self, num_users, embedding_dim=128, hidden_dim=256, num_heads=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Embedding layers
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim)  # +1 for padding
        self.spatial_embedding = SpatialEmbedding(embedding_dim=embedding_dim)
        self.temporal_embedding = TemporalEmbedding(embedding_dim=embedding_dim)
        
        # Special tokens
        self.mask_token = nn.Parameter(torch.randn(embedding_dim))
        
        # Spatial-temporal cross-attention
        self.spatial_temporal_attention = SpatialTemporalAttention(embedding_dim, num_heads=4, dropout=dropout)
        
        # Transformer blocks for sequential modeling
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embedding_dim * 3, num_heads, embedding_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        
        # LSTM for temporal sequence modeling
        self.lstm = nn.LSTM(
            input_size=embedding_dim * 3,  # user + spatial + temporal
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Multi-head attention for final sequence aggregation
        self.final_attention = MultiHeadAttention(hidden_dim * 2, num_heads=4, dropout=dropout)
        
        # Global attention mechanism for sequence-to-fixed
        self.global_attention = nn.Linear(hidden_dim * 2, 1)
        
        # Output heads
        self.user_classifier = nn.Linear(hidden_dim * 2, num_users)
        self.next_location_predictor = nn.Linear(hidden_dim * 2, 2)  # lat, lon
        
    def forward(self, user_ids, coordinates, timestamps, seq_lens, user_mask_prob=0.15):
        batch_size, seq_len = user_ids.shape
        
        # Create user mask for training
        user_mask = torch.zeros_like(user_ids, dtype=torch.bool)
        if self.training:
            # Randomly mask user_ids
            mask_indices = torch.rand(batch_size, seq_len) < user_mask_prob
            # Don't mask padding tokens
            for i, length in enumerate(seq_lens):
                mask_indices[i, length:] = False
            user_mask = mask_indices
        
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        # Apply masking
        user_emb[user_mask] = self.mask_token
        
        spatial_emb = self.spatial_embedding(coordinates)
        temporal_emb = self.temporal_embedding(timestamps)
        
        # Create padding mask for attention
        padding_mask = torch.zeros(batch_size, seq_len, device=user_ids.device)
        for i, length in enumerate(seq_lens):
            padding_mask[i, :length] = 1
        
        # Spatial-temporal cross-attention
        spatial_emb, temporal_emb = self.spatial_temporal_attention(
            spatial_emb, temporal_emb, mask=padding_mask
        )
        
        # Combine embeddings
        combined_emb = torch.cat([user_emb, spatial_emb, temporal_emb], dim=-1)
        
        # Pass through transformer blocks
        transformer_out = combined_emb
        all_attention_weights = []
        
        for transformer_block in self.transformer_blocks:
            transformer_out, attn_weights = transformer_block(transformer_out, mask=padding_mask)
            all_attention_weights.append(attn_weights)
        
        # LSTM encoding for temporal modeling
        lstm_out, (hidden, cell) = self.lstm(transformer_out)
        
        # Final multi-head attention
        attended_out, final_attn_weights = self.final_attention(lstm_out, mask=padding_mask)
        
        # Global attention mechanism for sequence-to-fixed
        global_attention_weights = F.softmax(self.global_attention(attended_out), dim=1)
        
        # Apply padding mask to global attention
        global_attention_weights = global_attention_weights * padding_mask.unsqueeze(-1)
        global_attention_weights = global_attention_weights / (global_attention_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Weighted sum for final representation
        context_vector = torch.sum(global_attention_weights * attended_out, dim=1)
        
        # Predictions
        user_pred = self.user_classifier(context_vector)
        next_location_pred = self.next_location_predictor(context_vector)
        
        return {
            'user_prediction': user_pred,
            'next_location_prediction': next_location_pred,
            'user_mask': user_mask,
            'attention_weights': global_attention_weights,
            'transformer_attention': all_attention_weights,
            'final_attention': final_attn_weights,
            'context_vector': context_vector
        }

class TrajectoryModel(nn.Module):
    """Complete trajectory model with hybrid transformer-LSTM architecture"""
    
    def __init__(self, num_users, embedding_dim=128, hidden_dim=256, num_heads=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.encoder = HybridTransformerLSTMEncoder(
            num_users, embedding_dim, hidden_dim, num_heads, num_layers, dropout
        )
        
    def forward(self, batch, user_mask_prob=0.15):
        return self.encoder(
            batch['user_ids'],
            batch['coordinates'], 
            batch['timestamps'],
            batch['seq_len'],
            user_mask_prob
        )
    
    def compute_loss(self, batch, outputs, alpha=1.0, beta=0.01):
        """Compute multi-task loss"""
        
        # User prediction loss (only on masked positions)
        user_mask = outputs['user_mask']
        if user_mask.sum() > 0:
            masked_user_ids = batch['user_ids'][user_mask]
            # Get predictions for each position, then select masked ones
            batch_size, seq_len = user_mask.shape
            user_preds_expanded = outputs['user_prediction'].unsqueeze(1).expand(-1, seq_len, -1)
            masked_user_preds = user_preds_expanded[user_mask]
            user_loss = F.cross_entropy(masked_user_preds, masked_user_ids)
        else:
            user_loss = torch.tensor(0.0, device=batch['user_ids'].device)
        
        # Next location prediction loss (simplified - predict last coordinate)
        # In practice, you'd want to predict next location given previous sequence
        batch_size = batch['coordinates'].shape[0]
        target_coords = []
        for i in range(batch_size):
            seq_len = batch['seq_len'][i]
            if seq_len > 1:
                # Use last coordinate as target
                target_coords.append(batch['coordinates'][i, seq_len-1])
            else:
                # Fallback for short sequences
                target_coords.append(batch['coordinates'][i, 0])
        
        target_coords = torch.stack(target_coords)
        location_loss = F.mse_loss(outputs['next_location_prediction'], target_coords)
        
        # Use location loss directly (scaling can be adjusted via beta parameter)
        total_loss = alpha * user_loss + beta * location_loss
        
        return {
            'total_loss': total_loss,
            'user_loss': user_loss,
            'location_loss': location_loss
        }


def train_model(learning_rate=0.001, batch_size=32, num_epochs=5, patience=10, 
                warmup_epochs=2, gradient_clip=1.0, weight_decay=1e-5,
                subset_fraction=1.0, max_batches_per_epoch=None):
    """Training loop with convergence improvements"""
    
    # Load data
    print("Loading trajectory data...")
    trajectories = load_data("C:\\Users\\dlfel\\Projects\\python\\mobility\\data\\cityD-dataset.csv")
    
    dataset = TrajectoryDataset(trajectories)  # num_users calculated automatically
    
    # Optionally use only a subset of the data
    if subset_fraction < 1.0:
        subset_size = int(len(dataset) * subset_fraction)
        indices = torch.randperm(len(dataset))[:subset_size]
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"Using subset of {len(dataset)} trajectories ({subset_fraction*100:.1f}% of data)")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = TrajectoryModel(num_users=4000)
    
    # Improved optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience//2
    )
    
    # Warmup scheduler for first few epochs
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    
    print(f"Training with 4000 users, {len(dataset)} trajectories")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    
    # Early stopping variables
    best_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Stop early if max_batches_per_epoch is reached
            if max_batches_per_epoch is not None and batch_idx >= max_batches_per_epoch:
                print(f"Reached max batches per epoch ({max_batches_per_epoch}), stopping epoch early")
                break
                
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch, user_mask_prob=0.15)
            
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
            num_batches += 1
            
            if batch_idx % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                      f"Loss: {losses['total_loss']:.4f}, "
                      f"User Loss: {losses['user_loss']:.4f}, "
                      f"Location Loss: {losses['location_loss']:.4f}, "
                      f"LR: {current_lr:.6f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        print(f"Epoch {epoch} completed, Average Loss: {avg_loss:.4f}")
        
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
                'num_users': 4000
            }
            torch.save(best_checkpoint, 'best_model.pth')
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
            'loss': total_loss/len(dataloader),
            'num_users': 4000
        }
        torch.save(checkpoint, f'model_epoch_{epoch}.pth')
        print(f"Model saved as model_epoch_{epoch}.pth")

def load_trained_model(checkpoint_path):
    """Load a trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model = TrajectoryModel(num_users=checkpoint['num_users'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint

def create_model_variants():
    """Create different model variants for experimentation"""
    variants = {
        'light': TrajectoryModel(num_users=4000, embedding_dim=64, hidden_dim=128, 
                                num_heads=4, num_layers=2, dropout=0.1),
        'standard': TrajectoryModel(num_users=4000, embedding_dim=128, hidden_dim=256, 
                                   num_heads=8, num_layers=3, dropout=0.1),
        'heavy': TrajectoryModel(num_users=4000, embedding_dim=256, hidden_dim=512, 
                                num_heads=16, num_layers=4, dropout=0.1)
    }
    return variants

def embed_new_user(model, user_trajectories, method='mean'):
    """
    Project a new user into the embedding space using their trajectories
    
    Args:
        model: trained TrajectoryModel
        user_trajectories: list of trajectories for the new user
        method: 'mean' or 'attention' - how to aggregate trajectory embeddings
    
    Returns:
        user_embedding: tensor of shape [embedding_dim]
    """
    model.eval()
    trajectory_embeddings = []
    
    with torch.no_grad():
        for traj in user_trajectories:
            # Create a dummy dataset with just this trajectory
            # Replace user_id with a placeholder (we'll ignore user embeddings)
            traj_with_dummy_user = [(0, *point[1:]) for point in traj]
            
            # Convert to tensor format
            dataset = TrajectoryDataset([traj_with_dummy_user], max_length=50)
            batch = dataset[0]
            
            # Add batch dimension
            for key in batch:
                batch[key] = batch[key].unsqueeze(0)
            
            # Get spatial and temporal embeddings (ignore user embeddings for new user)
            spatial_emb = model.encoder.spatial_embedding(batch['coordinates'])
            temporal_emb = model.encoder.temporal_embedding(batch['timestamps'])
            
            # Combine spatial and temporal only
            combined_emb = torch.cat([spatial_emb, temporal_emb], dim=-1)
            
            # Pass through LSTM
            lstm_out, _ = model.encoder.lstm(combined_emb)
            
            # Apply attention to get trajectory representation
            attention_weights = F.softmax(model.encoder.attention(lstm_out), dim=1)
            seq_len = batch['seq_len'].item()
            
            # Mask padding
            padding_mask = torch.zeros(1, lstm_out.shape[1])
            padding_mask[0, :seq_len] = 1
            attention_weights = attention_weights * padding_mask.unsqueeze(-1)
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
            
            # Get trajectory embedding
            traj_embedding = torch.sum(attention_weights * lstm_out, dim=1)
            trajectory_embeddings.append(traj_embedding.squeeze(0))
    
    # Aggregate trajectory embeddings
    if method == 'mean':
        user_embedding = torch.stack(trajectory_embeddings).mean(dim=0)
    elif method == 'attention':
        # Use learned attention to weight trajectory embeddings
        embeddings_stack = torch.stack(trajectory_embeddings)
        weights = F.softmax(torch.randn(len(trajectory_embeddings)), dim=0)
        user_embedding = torch.sum(weights.unsqueeze(-1) * embeddings_stack, dim=0)
    
    return user_embedding

def embed_trajectory(model, trajectory, user_embedding=None):
    """
    Project a trajectory into the embedding space
    
    Args:
        model: trained TrajectoryModel
        trajectory: single trajectory to embed
        user_embedding: optional user embedding to use instead of learned user embeddings
    
    Returns:
        trajectory_embedding: tensor of shape [hidden_dim*2]
        attention_weights: attention weights for each position
    """
    model.eval()
    
    with torch.no_grad():
        # Prepare trajectory data
        if user_embedding is not None:
            # Use provided user embedding
            traj_with_dummy_user = [(0, *point[1:]) for point in trajectory]
        else:
            # Use original user IDs
            traj_with_dummy_user = trajectory
            
        dataset = TrajectoryDataset([traj_with_dummy_user], max_length=50)
        batch = dataset[0]
        
        # Add batch dimension
        for key in batch:
            batch[key] = batch[key].unsqueeze(0)
        
        if user_embedding is not None:
            # Replace user embeddings with provided embedding
            spatial_emb = model.encoder.spatial_embedding(batch['coordinates'])
            temporal_emb = model.encoder.temporal_embedding(batch['timestamps'])
            
            # Repeat user embedding for sequence length
            seq_len = batch['coordinates'].shape[1]
            user_emb = user_embedding.unsqueeze(0).unsqueeze(0).repeat(1, seq_len, 1)
            
            combined_emb = torch.cat([user_emb, spatial_emb, temporal_emb], dim=-1)
        else:
            # Use model's forward pass
            outputs = model(batch, user_mask_prob=0.0)  # No masking for inference
            return outputs['context_vector'].squeeze(0), outputs['attention_weights'].squeeze(0)
        
        # Continue with LSTM and attention
        lstm_out, _ = model.encoder.lstm(combined_emb)
        attention_weights = F.softmax(model.encoder.attention(lstm_out), dim=1)
        
        # Apply sequence length mask
        seq_len = batch['seq_len'].item()
        padding_mask = torch.zeros(1, lstm_out.shape[1])
        padding_mask[0, :seq_len] = 1
        attention_weights = attention_weights * padding_mask.unsqueeze(-1)
        attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Get trajectory embedding
        trajectory_embedding = torch.sum(attention_weights * lstm_out, dim=1)
        
        return trajectory_embedding.squeeze(0), attention_weights.squeeze(0)

def find_similar_users(model, new_user_embedding, existing_user_ids, top_k=5):
    """
    Find most similar existing users to a new user
    
    Args:
        model: trained model
        new_user_embedding: embedding of new user
        existing_user_ids: list of existing user IDs
        top_k: number of similar users to return
    
    Returns:
        similar_users: list of (user_id, similarity_score) tuples
    """
    similarities = []
    
    with torch.no_grad():
        for user_id in existing_user_ids:
            # Get existing user embedding
            existing_embedding = model.encoder.user_embedding(torch.tensor([user_id]))
            
            # Compute cosine similarity
            similarity = F.cosine_similarity(
                new_user_embedding.unsqueeze(0), 
                existing_embedding, 
                dim=1
            ).item()
            
            similarities.append((user_id, similarity))
    
    # Sort by similarity and return top k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# Alternative training strategies for convergence issues
def train_with_strategy(strategy='default'):
    """
    Train with different strategies for convergence issues
    
    Strategies:
    - 'default': Standard training
    - 'conservative': Lower LR, smaller batch, more regularization
    - 'aggressive': Higher LR, larger batch, less regularization  
    - 'fast_epochs': Use subset of data per epoch
    - 'limited_batches': Limit batches per epoch
    - 'curriculum': Start with shorter sequences, gradually increase
    """
    
    if strategy == 'conservative':
        return train_model(
            learning_rate=0.0001,
            batch_size=16, 
            num_epochs=20,
            patience=15,
            gradient_clip=0.5,
            weight_decay=1e-4
        )
    elif strategy == 'aggressive':
        return train_model(
            learning_rate=0.01,
            batch_size=64,
            num_epochs=10,
            patience=5,
            gradient_clip=2.0,
            weight_decay=1e-6
        )
    elif strategy == 'fast_epochs':
        # Use only 20% of data per epoch, but more epochs
        return train_model(
            subset_fraction=0.2,
            num_epochs=25,
            patience=15
        )
    elif strategy == 'limited_batches':
        # Limit to 500 batches per epoch
        return train_model(
            max_batches_per_epoch=500,
            num_epochs=15
        )
    elif strategy == 'curriculum':
        # Start with shorter sequences
        print("Phase 1: Training with max_length=20")
        train_curriculum_phase(max_length=20, epochs=5)
        print("Phase 2: Training with max_length=35") 
        train_curriculum_phase(max_length=35, epochs=5)
        print("Phase 3: Training with max_length=50")
        return train_model(num_epochs=10)
    else:
        return train_model()

def train_curriculum_phase(max_length, epochs):
    """Helper for curriculum learning"""
    trajectories = load_data("C:\\Users\\dlfel\\Projects\\python\\mobility\\data\\cityD-dataset.csv")
    dataset = TrajectoryDataset(trajectories, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = TrajectoryModel(num_users=4000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(batch, user_mask_prob=0.15)
            losses = model.compute_loss(batch, outputs)
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += losses['total_loss'].item()
        
        print(f"Curriculum Phase - Epoch {epoch}, Avg Loss: {total_loss/len(dataloader):.4f}")

if __name__ == "__main__":
    # Choose training strategy based on your needs:
    # train_with_strategy('default')        # Standard training
    # train_with_strategy('conservative')   # If loss explodes or doesn't decrease
    # train_with_strategy('aggressive')     # If training is too slow
    # train_with_strategy('curriculum')     # If model struggles with long sequences
    train_with_strategy('fast_epochs')    # Use 20% of data per epoch (faster epochs)
    # train_with_strategy('limited_batches') # Limit to 500 batches per epoch
    
    # Or use custom parameters:
    # train_model(subset_fraction=0.1, max_batches_per_epoch=100)  # Very fast epochs
    # train_model(subset_fraction=0.5)  # Use 50% of data
    
    # train_model()