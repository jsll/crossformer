import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

def plot_attention_rollout(
    model,
    observations: dict,
    tasks: dict,
    readout_name: str,
    save_path: str = None,
) -> plt.Figure:
    """
    Visualizes attention rollout from readout tokens to input tokens.
    
    Args:
        model: CrossFormerModel instance
        observations: Dictionary of observations
        tasks: Dictionary of task specifications
        readout_name: Name of readout head to analyze (e.g. "readout_single_arm")
        save_path: Optional path to save visualization
    Returns:
        matplotlib Figure
    """
    # Get attention rollout and token mapping
    rollout, token_map = model.analyze_attention(observations, tasks, readout_name.replace("readout_", ""))
    
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    
    # Plot images with attention overlay
    n_timesteps = observations["timestep_pad_mask"].shape[1]
    for t in range(n_timesteps):
        plt.subplot(1, n_timesteps, t+1)
        
        # Get image tokens for this timestep
        obs_token_idxs = [i for i, name in token_map.items() if name.startswith("obs_")]
        obs_attention = rollout[t, obs_token_idxs]
        
        # Reshape attention to match image grid
        grid_size = int(np.sqrt(len(obs_token_idxs) // len(model.config["model"]["observation_tokenizers"])))
        attention_grid = obs_attention.reshape(grid_size, grid_size)
        
        # Get image for this timestep
        for k,v in observations.items():
            if k.startswith("image_"):
                img = v[0,t]
                break
        
        # Plot image
        plt.imshow(img)
        
        # Overlay attention heatmap
        attention_resized = jnp.array(plt.mpl.transforms.resize(attention_grid, img.shape[:2]))
        plt.imshow(attention_resized, cmap='hot', alpha=0.5)
        plt.axis('off')
        plt.title(f'Timestep {t}')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    
    return fig

def plot_readout_attention(
    model,
    observations: dict,
    tasks: dict,
    readout_name: str,
    save_path: str = None,
) -> plt.Figure:
    """
    Plots attention weights from readout tokens as a heatmap.
    
    Args:
        model: CrossFormerModel instance  
        observations: Dictionary of observations
        tasks: Dictionary of task specifications
        readout_name: Name of readout head to analyze
        save_path: Optional path to save visualization
    Returns:
        matplotlib Figure
    """
    # Get attention rollout and token mapping
    rollout, token_map = model.analyze_attention(observations, tasks, readout_name.replace("readout_", ""))
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(rollout, cmap='viridis')
    
    # Add token labels
    token_labels = list(token_map.values())
    ax.set_xticks(np.arange(len(token_labels)))
    ax.set_yticks(np.arange(len(token_labels)))
    ax.set_xticklabels(token_labels, rotation=45, ha='right')
    ax.set_yticklabels(token_labels)
    
    # Add colorbar
    plt.colorbar(im)
    
    plt.title(f'Attention Weights from {readout_name}')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    return fig
