import torch
import numpy as np
from train_s2s import TrajectoryModel, TrajectoryDataset
import pandas as pd
from itertools import pairwise
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pickle

def load_data(file_path):    
  df = pd.read_csv(file_path)
  df = df[df['uid'] > 4000]
  return df_to_list(df)


def df_to_list(df, normalize=True):
  def is_weekend(day):
    # 0 sunday, 1 monday, ..., 6 saturday
    temp = day % 7
    return int(temp == 0 or temp == 6)
  
  if normalize:
      scale = 200
  else:
      scale = 1
  data = []
  for _, g in df.groupby(['uid', 'd']):
    if len(g) < 2:
        continue
    temp = g.to_records(index=False).tolist()
    temp2 = list(map(lambda x: (x[0], x[3], x[4], is_weekend(x[1]), x[2]), temp))
    temp3 = pairwise(temp2)
    temp4 = list(map(lambda x: (x[0][0], x[0][1]/scale, x[0][2]/scale, x[0][3], x[0][4], x[1][4]), temp3))
    data.append(temp4)
  return data
    

def load_best_model(checkpoint_path='best_model.pth'):
    """
    Load the best trained model from checkpoint
    
    Args:
        checkpoint_path: Path to the saved model checkpoint
        
    Returns:
        model: Loaded TrajectoryModel in evaluation mode
        checkpoint: Checkpoint dictionary with training info
    """
    print(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model with same architecture as training
    # Note: Adjust these parameters to match your trained model
    model = TrajectoryModel(
        embedding_dim=128, 
        hidden_dim=256, 
        num_heads=8, 
        num_layers=3, 
        dropout=0.1
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    print(f"Model loaded successfully!")
    print(f"Training epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"Training loss: {checkpoint.get('loss', 'Unknown'):.6f}")
    
    return model, checkpoint

def get_trajectory_embedding(model, trajectory):
    """
    Get spatio-temporal embedding for a new trajectory
    
    Args:
        model: Trained TrajectoryModel
        trajectory: List of trajectory points in format:
                   [(user_id, lat, lon, is_weekend, timestamp, next_timestamp), ...]
                   
    Returns:
        embedding: Trajectory embedding vector of shape [hidden_dim*2]
        attention_weights: Attention weights showing important parts
        predicted_next_location: Model's prediction for next location
        actual_next_location: Actual next location (if available)
    """
    model.eval()
    
    with torch.no_grad():
        # Convert trajectory to dataset format (no augmentation for inference)
        dataset = TrajectoryDataset([trajectory], max_length=50, training=False)
        batch = dataset[0]
        
        # Store actual target for comparison
        actual_next = batch['target_coords'].clone()
        
        # Add batch dimension for model input
        for key in batch:
            if key != 'target_coords':  # Don't add batch dim to target
                batch[key] = batch[key].unsqueeze(0)
        
        # Forward pass through model
        outputs = model(batch)
        
        # Extract results
        embedding = outputs['context_vector'].squeeze(0)
        attention_weights = outputs['attention_weights'].squeeze(0)
        predicted_next = outputs['next_location_prediction'].squeeze(0)
        
        return embedding, attention_weights, predicted_next, actual_next

def analyze_trajectory(model, trajectory, verbose=True):
    """
    Comprehensive analysis of a trajectory
    
    Args:
        model: Trained TrajectoryModel
        trajectory: Trajectory to analyze
        verbose: Whether to print detailed analysis
        
    Returns:
        analysis: Dictionary with analysis results
    """
    embedding, attention, predicted, actual = get_trajectory_embedding(model, trajectory)
    
    # Calculate prediction error
    prediction_error = torch.norm(predicted - actual).item()
    
    # Get most attended positions
    attention_np = attention.cpu().numpy().flatten()
    top_attention_indices = np.argsort(attention_np)[-3:][::-1]  # Top 3 most attended
    
    analysis = {
        'embedding': embedding,
        'embedding_norm': torch.norm(embedding).item(),
        'predicted_location': predicted.tolist(),
        'actual_location': actual.tolist(),
        'prediction_error': prediction_error,
        'prediction_error_km': prediction_error * 200 * 111,  # Rough conversion to km
        'attention_weights': attention,
        'top_attention_positions': top_attention_indices.tolist(),
        'trajectory_length': len(trajectory)
    }
    
    if verbose:
        print("\n" + "="*50)
        print("TRAJECTORY ANALYSIS")
        print("="*50)
        print(f"Trajectory length: {analysis['trajectory_length']} points")
        print(f"Embedding dimension: {embedding.shape[0]}")
        print(f"Embedding norm: {analysis['embedding_norm']:.4f}")
        print(f"\nPredicted next location: [{predicted[0]:.4f}, {predicted[1]:.4f}]")
        print(f"Actual next location:    [{actual[0]:.4f}, {actual[1]:.4f}]")
        print(f"Prediction error: {prediction_error:.6f} (normalized)")
        print(f"Prediction error: ~{analysis['prediction_error_km']:.1f} km")
        print(f"\nMost attended positions: {top_attention_indices}")
        print("="*50)
    
    return analysis


def example_usage():
    """Example of how to use the inference functions"""
    def get_embedding(model):
        def foo(trajectory):
            embedding, _, _, _ = get_trajectory_embedding(model, trajectory)
            return embedding.cpu().numpy()
        return foo
    
    # Load the trained model
    try:
        model, _ = load_best_model('best_model.pth')
        helper_func = get_embedding(model)
    except FileNotFoundError:
        print("Error: best_model.pth not found. Please train a model first.")
        return
    
    # Load some sample data
    print("\nLoading trajectories...")
    file_path="C:\\Users\\dlfel\\Projects\\python\\mobility\\data\\cityD-dataset.csv"
    df = pd.read_csv(file_path)
    df = df[df['uid'] < 1000]
    v = []
    labels = []
    traj = []
    for i,g in tqdm(df.groupby('uid')):
        traj.extend(df_to_list(g,False))
        data = df_to_list(g)
        labels.extend([i]*len(data))
        embeddings = [helper_func(t) for t in data]
        v.extend(embeddings)

    pca_results = get_pca_projection(v)

    pca_2d = pca_results['pca_2d']
    
    # Save variables to results folder
    os.makedirs('results', exist_ok=True)
    
    # Save trajectories
    with open('results/traj.pkl', 'wb') as f:
        pickle.dump(traj, f)
    
    # Save labels
    with open('results/labels.pkl', 'wb') as f:
        pickle.dump(labels, f)
    
    # Save PCA 2D projections
    np.save('results/pca_2d.npy', pca_2d)
    
    print("Saved variables to results folder:")
    print(f"  - traj.pkl ({len(traj)} trajectories)")
    print(f"  - labels.pkl ({len(labels)} labels)")
    print(f"  - pca_2d.npy ({pca_2d.shape})")
    
    # Plot the first two principal components
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_results['pca_2d'][:, 0], pca_results['pca_2d'][:, 1], 
                         c=labels, cmap='viridis', alpha=0.6, s=30)
    plt.colorbar(scatter, label='User ID')
    plt.xlabel(f'First Principal Component ({pca_results["variance_explained"][0]*100:.1f}% variance)')
    plt.ylabel(f'Second Principal Component ({pca_results["variance_explained"][1]*100:.1f}% variance)')
    plt.title('Trajectory Embeddings - First Two Principal Components (Colored by User ID)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pca_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Saved plot to pca_scatter_plot.png")
    print(f"First two components explain {sum(pca_results['variance_explained'][:2])*100:.1f}% of variance")

    
    

def get_pca_projection(embeddings):
    # Convert list of embeddings to matrix
    embedding_matrix = np.array(embeddings)
    
    # Normalize the matrix using scikit-learn
    scaler = StandardScaler(with_std=False)
    standardized_embeddings = scaler.fit_transform(embedding_matrix)
    
    # Apply PCA
    pca = PCA()
    pca_embeddings = pca.fit_transform(standardized_embeddings)
    
    return {
        'pca_2d': pca_embeddings[:, :2],  # First two components
        'pca_full': pca_embeddings,
        'variance_explained': pca.explained_variance_ratio_,
        'pca_model': pca
    }


if __name__ == "__main__":
    print("Trajectory Embedding Inference Tool")
    print("====================================")
    
    # Run example usage
    example_usage()