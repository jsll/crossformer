import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import cv2
import matplotlib.gridspec as gridspec

def get_observation_image(observation, observation_type):
    image=None
    for k in observation.keys():
        if observation_type in k:
            image = observation[k].squeeze()
            break
    return image

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = plt.cm.jet(mask, cv2.COLORMAP_JET)[:,:,:3]
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


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
    print(indexes_obs)
    # Create a grid of subplots: num_images rows × num_timesteps columns
    fig, axs = plt.subplots(1, num_timesteps+2,squeeze=False) 
                           #figsize=(4*num_timesteps, 4*num_images),squeeze=False)

    if observation_image is None:
      observation_image = get_observation_image(observations, observation_type)
    im = axs[0, 0].imshow(observation_image, cmap='viridis')

    # If there's only one image, wrap axs in a list to make it 2D
    if num_images == 1:
        axs = np.array([axs])
    patch_size = 32
    print(indexes_obs)

    for t in range(num_timesteps):
        rollout = rollouts[t]
        # Create heatmap
    
        images_per_readout = []
        for num_image in range(num_images):
            print(indexes_readout[num_image])
            print("asd")
            image = np.zeros((224,224))
            x=0
            y=0
            for index_obs in indexes_obs:
                attention = rollout[indexes_readout[num_image], index_obs]
                image[x:x+patch_size, y:y+patch_size] = attention
                x+= patch_size
                if x== 224:
                    x=0
                    y+=patch_size
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
        axs[0, t+2].imshow(rollout,cmap='viridis')

        axs[0,t+1].axis('off')  # Remove axes
            
        #axs[0, t+1].set_title(f'Timestep {t}')

        axs[0, t].set_title(title)
        axs[0, t+1].set_title(f'Attention Rollout')
 
    
        
        
    # Add a colorbar that applies to all subplots
    fig.colorbar(im, ax=axs.ravel().tolist())
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)

    return fig

def plot_readout_attention2(
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
    # Create a grid of subplots: num_images rows × num_timesteps columns
    #fig, axs = plt.subplots(1, num_timesteps+2,figsize=(10,10),squeeze=False) 
                           #figsize=(4*num_timesteps, 4*num_images),squeeze=False)
    # Create figure with GridSpec
    fig = plt.figure(figsize=(15, 5))
    # Create a nested GridSpec
    outer_grid = plt.GridSpec(1, 3, width_ratios=[1, 1, 1.1])  # Slightly wider last column for colorbar
    axs = []
    for i in range(num_timesteps + 2):
        axs.append(fig.add_subplot(outer_grid[i]))


    if observation_image is None:
      observation_image = get_observation_image(observations, observation_type)

    # If there's only one image, wrap axs in a list to make it 2D
    if num_images == 1:
        axs = np.array([axs])

    for t in range(num_timesteps):
        rollout = rollouts[t]
    
        images_per_readout = []
        for num_image in range(num_images):
            image = np.zeros((224,224))
            mask = rollout[indexes_readout[num_image], indexes_obs]
            mask = np.asarray(mask.reshape(7,7))
            mask = mask / np.max(mask)
            mask = cv2.resize(mask, (224, 224))
            images_per_readout.append(mask.copy())

        average_readout_image = np.asarray(images_per_readout).mean(0)

        
        overlay = show_mask_on_image(observation_image, average_readout_image)
        # Create a dark overlay
        """  
        axs[0, t].imshow(observation_image, cmap='jet')

        axs[0,t].axis('off')  # Remove axes
      
        axs[0, t+1].imshow(overlay,cmap='jet')

        #axs[0, t+1].imshow(dark_overlay, alpha=alpha, cmap='gray')
        im = axs[0, t+2].imshow(average_readout_image,cmap='jet')

        axs[0,t+1].axis('off')  # Remove axes
        axs[0,t+2].axis('off')  # Remove axes
 
        axs[0, t].set_title(title)
        axs[0, t+1].set_title(f'Attention superimposed on image')
        axs[0, t+2].set_title(f'Attention Rollout')
        """
      
        # Plot on the GridSpec subplots
        axs[t].imshow(observation_image, cmap='jet')
        axs[t].axis('off')
        
        axs[t+1].imshow(overlay, cmap='jet')
        axs[t+1].axis('off')
        
        im = axs[t+2].imshow(average_readout_image, cmap='jet')
        axs[t+2].axis('off')
 
        axs[t].set_title(title)
        axs[t+1].set_title('Attention superimposed on image')
        axs[t+2].set_title('Attention Rollout')

    # Add colorbar with specific spacing
    plt.colorbar(im, ax=axs[t+2], fraction=0.046, pad=0.04)

    #from mpl_toolkits.axes_grid1 import make_axes_locatable
    #divider = make_axes_locatable(axs[t+2])
    #cax = divider.append_axes("right", size="5%", pad=0.1)
    #plt.colorbar(im, cax=cax)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    return fig

