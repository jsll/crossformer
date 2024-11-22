import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

def get_observation_image(observation, observation_type):
    image=None
    for k in observation.keys():
        if observation_type in k:
            image = observation[k].squeeze()
            break
    return image

def plot_readout_attention(
    rollouts,
    token_types,
    head,
    observations,
    observation_type = "_high",
    observation_image=None,
    save_path: str = None,
    title = ""
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
    indexes_readout = []
    indexes_obs = []
    for i, j in enumerate(token_types):
        if j == head:
            indexes_readout.append(i)
        if observation_type in j:
            indexes_obs.append(i)
    num_timesteps =  rollouts.shape[0]
    num_images = len(indexes_readout)
    print(indexes_readout)
    # Create a grid of subplots: num_images rows × num_timesteps columns
    fig, axs = plt.subplots(1, num_timesteps+1,squeeze=False) 
                           #figsize=(4*num_timesteps, 4*num_images),squeeze=False)

    if observation_image is None:
      observation_image = get_observation_image(observations, observation_type)
    im = axs[0, 0].imshow(observation_image, cmap='viridis')

    # If there's only one image, wrap axs in a list to make it 2D
    if num_images == 1:
        axs = np.array([axs])

    for t in range(num_timesteps):
        rollout = rollouts[t]
        # Create heatmap
    
        images_per_readout = []
        for num_image in range(num_images):
            image = np.zeros((224,224))
            x=0
            y=0
            for index_obs in indexes_obs:
                attention = rollout[indexes_readout[num_image], index_obs]
                image[x:x+32, y:y+32] = attention
                x+= 32
                if x== 224:
                    x=0
                    y+=32
            images_per_readout.append(image.copy())
            # Plot the image in the corresponding subplot
        average_readout_image = np.asarray(images_per_readout).mean(0)
        # Plot original image
        axs[0, t+1].imshow(observation_image)
        # Overlay attention map with alpha
        alpha = average_readout_image / average_readout_image.max()  # Normalize to [0,1]s
        # We subtract alpha from 1 as the higher the attention the closer to zero should the alpha value be as a
        # value of 0 means transparent and 1 means opaque.

        alpha = 1-alpha
        
        # Create a dark overlay

        dark_overlay = np.zeros((observation_image.shape[0], observation_image.shape[1],1))
        axs[0,t].axis('off')  # Remove axes

        axs[0, t+1].imshow(dark_overlay, alpha=alpha, cmap='gray')

        axs[0,t+1].axis('off')  # Remove axes
            
        #axs[0, t+1].set_title(f'Timestep {t}')

        axs[0, t].set_title(title)
        axs[0, t+1].set_title(f'Attention Rollout')
 
    
        
        
    # Add a colorbar that applies to all subplots
    #fig.colorbar(im, ax=axs.ravel().tolist())
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)

    return fig
