import torch
import numpy as np
from train_s2s_autoencoder import TrajectoryAutoencoderModel, TrajectoryAutoencoderDataset, load_trained_autoencoder, encode_trajectory
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
    

def load_best_model(checkpoint_path='best_autoencoder.pth'):
    """
    Load the best trained autoencoder from checkpoint
    
    Args:
        checkpoint_path: Path to the saved model checkpoint
        
    Returns:
        model: Loaded TrajectoryAutoencoderModel in evaluation mode
        checkpoint: Checkpoint dictionary with training info
    """
    print(f"Loading autoencoder from {checkpoint_path}")
    
    # Use the function from train_s2s_autoencoder
    model, checkpoint = load_trained_autoencoder(checkpoint_path)
    
    print(f"Autoencoder loaded successfully!")
    print(f"Training epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"Training loss: {checkpoint.get('loss', 'Unknown'):.6f}")
    
    return model, checkpoint

def get_trajectory_embedding(model, trajectory):
    """
    Get encoder embedding for a trajectory using autoencoder
    
    Args:
        model: Trained TrajectoryAutoencoderModel
        trajectory: List of trajectory points in format:
                   [(user_id, lat, lon, is_weekend, timestamp, next_timestamp), ...]
                   
    Returns:
        embedding: Trajectory latent representation from encoder
        encoder_outputs: Full encoder sequence outputs
    """
    # Use the encode_trajectory function from train_s2s_autoencoder
    latent_representation, encoder_outputs = encode_trajectory(model, trajectory)
    
    return latent_representation, encoder_outputs

def analyze_trajectory(model, trajectory, verbose=True):
    """
    Analysis of a trajectory using autoencoder
    
    Args:
        model: Trained TrajectoryAutoencoderModel
        trajectory: Trajectory to analyze
        verbose: Whether to print detailed analysis
        
    Returns:
        analysis: Dictionary with analysis results
    """
    embedding, encoder_outputs = get_trajectory_embedding(model, trajectory)
    
    analysis = {
        'embedding': embedding,
        'embedding_norm': torch.norm(embedding).item(),
        'encoder_outputs': encoder_outputs,
        'trajectory_length': len(trajectory)
    }
    
    if verbose:
        print("\n" + "="*50)
        print("TRAJECTORY ANALYSIS (AUTOENCODER)")
        print("="*50)
        print(f"Trajectory length: {analysis['trajectory_length']} points")
        print(f"Latent embedding dimension: {embedding.shape[0]}")
        print(f"Embedding norm: {analysis['embedding_norm']:.4f}")
        print("="*50)
    
    return analysis


def example_usage():
    """Example of how to use the autoencoder inference functions"""
    def get_embedding(model):
        def foo(trajectory):
            embedding, _ = get_trajectory_embedding(model, trajectory)
            return embedding.cpu().numpy()
        return foo
    
    # Load the trained autoencoder
    try:
        model, _ = load_best_model('best_autoencoder.pth')
        helper_func = get_embedding(model)
    except FileNotFoundError:
        print("Error: best_autoencoder.pth not found. Please train an autoencoder first.")
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
    
    # Save PCA model and full results
    with open('results/pca_results.pkl', 'wb') as f:
        pickle.dump(pca_results, f)
    
    print("Saved variables to results folder:")
    print(f"  - traj.pkl ({len(traj)} trajectories)")
    print(f"  - labels.pkl ({len(labels)} labels)")
    print(f"  - pca_2d.npy ({pca_2d.shape})")
    print(f"  - pca_results.pkl (PCA model and full results)")
    
    # Plot the first two principal components
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_results['pca_2d'][:, 0], pca_results['pca_2d'][:, 1], 
                         c=labels, cmap='viridis', alpha=0.6, s=30)
    plt.colorbar(scatter, label='User ID')
    plt.xlabel(f'First Principal Component ({pca_results["variance_explained"][0]*100:.1f}% variance)')
    plt.ylabel(f'Second Principal Component ({pca_results["variance_explained"][1]*100:.1f}% variance)')
    plt.title('Autoencoder Trajectory Embeddings - PCA 2D Projection (Colored by User ID)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/autoencoder_pca_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Saved plot to results/autoencoder_pca_scatter.png")
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