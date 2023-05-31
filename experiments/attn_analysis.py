import matplotlib.pyplot as plt
import matplotlib.animation as animation
import collections
import jax.numpy as jnp
import numpy as np

FONTSIZE = 14

def array_from_fig(fig):
  # Save the figure to a numpy array
  fig.canvas.draw()
  image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  img = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close(fig)
  return img

# Define a function that updates the plot for each frame of the video
def make_animation(images, slot_attns, display=True,  figsize=(8, 8), **kwargs):
    nslots = slot_attns.shape[1]
    
    fig, axs = plt.subplots(1, 1+nslots,  figsize=figsize)
    def update(idx):
        # Load the image + attention
        image = images[idx]
        slot_attn = slot_attns[idx]

        # Clear each axis
        for ax in axs:
            ax.clear()

        # plot attention
        plot_all_attn(image, slot_attn, axs=axs, **kwargs)

    # Create an animation object using the FuncAnimation() function
    anim = animation.FuncAnimation(fig, update, frames=len(images), interval=1000)
    return anim

def make_matrix(images, slot_attns,
                base_width=2,
                figsize=None,
                im_only: bool = False,
                time_with_x: bool = False,
                slot_titles=None,
                vmin_pre = None,
                vmax_pre = None, 
                img_title='Task',
                fontsize=16,
                shared_min_max: str = 'global',
                **kwargs):
    nslots = slot_attns.shape[1]
    ntimesteps = len(images)
    assert shared_min_max in ('global', 'timestep', 'none')
    if figsize is None:
        slot_size = (nslots + 1)*base_width
        time_size = ntimesteps*base_width
        if time_with_x:
          figsize = (time_size, slot_size)
        else:
          figsize = (slot_size, time_size)
        # print("Make figure", figsize)

    if time_with_x:
      fig, all_axs = plt.subplots(1+nslots, ntimesteps, figsize=figsize)
    else:
       fig, all_axs = plt.subplots(ntimesteps, 1+nslots, figsize=figsize)
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    if not im_only:
      if time_with_x:
        all_axs[0, 0].set_ylabel("image", fontsize=fontsize)
      else:
        for t in range(ntimesteps):
            all_axs[t, 0].set_ylabel(f"t={t+1}", fontsize=fontsize)

    if time_with_x:
      if ntimesteps == 1:
        all_axs = all_axs[:, None]
      axs = all_axs[:, 0]
    else:
      if ntimesteps == 1:
        all_axs = all_axs[None, :]
      axs = all_axs[0]

    if not im_only:
      axs[0].set_title(img_title, fontsize=fontsize)

      if slot_titles:
        for idx in range(nslots):
          if time_with_x:
            # set ylabel of first column to slot names
            axs[idx+1].set_ylabel(slot_titles[idx], fontsize=fontsize)
          else:
            # set title of first row to slot names
            axs[idx+1].set_title(slot_titles[idx], fontsize=fontsize)

    def update(uidx):
        # Load the image + attention
        image = images[uidx]
        slot_attn = slot_attns[uidx]
        if time_with_x:
          axs = all_axs[:, uidx]
        else:
          axs = all_axs[uidx]

        vmin = vmin_pre
        vmax = vmax_pre
        if shared_min_max == 'global':
           vmin = vmin_pre or slot_attns.min()
           vmax = vmax_pre or slot_attns.max()
        elif shared_min_max == 'timestep':
           vmin = vmin_pre or slot_attn.min()
           vmax = vmax_pre or slot_attn.max()
        elif shared_min_max == 'none':
           pass
        # plot attention
        plot_all_attn(image, slot_attn,
                      axs=axs,
                      vmin=vmin, vmax=vmax,
                      im_only=im_only, **kwargs)

    for t in range(ntimesteps):
        update(t)

    return fig, all_axs

def plot_all_attn(image,
    slot_attn,
    axs,
    img_title=None,
    slot_titles=None,
    im_only=False,
    shared_min_max=False,
    vmin=None,
    vmax=None):
    nslots = slot_attn.shape[0]
    assert len(axs) == nslots + 1 # including image

    if shared_min_max:
      # share across all slots
      vmin = slot_attn.min()
      vmax = slot_attn.max()

    axs[0].imshow(image)

    if not im_only and img_title:
        axs[0].set_title(img_title)
    for idx in range(nslots):
        attn = slot_attn[idx]
        if not im_only and slot_titles:
            axs[idx+1].set_title(slot_titles[idx])
        ax_out = plot_heatmap(im=image,
             h=attn,
             resize=8,
             cmap='gist_gray',
             ax=axs[idx+1],
             show=False)
        if vmin is not None and vmax is not None:
            ax_out.set_clim(vmin=vmin, vmax=vmax)

    # Remove the x and y ticks from the subplots
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

# function for heatmap
def plot_heatmap(im, h, resize=8, cmap='jet', ax=None, show=False):
    if ax is None:
        fig, ax = plt.subplots()
    h = np.kron(h, np.ones((resize, resize)))
    ax_out = ax.imshow(im)
    ax_out = ax.imshow(h, cmap=cmap, alpha=0.5, interpolation='bilinear')
    if show:
        plt.show()
    return ax_out


def timestep_img_attn(image, img_attn, **kwargs):
  # image: H, W, C
  # img_attn: N, H, W
  fig, _ = make_matrix(
      images=image[None],
      slot_attns=img_attn[None],
      **kwargs
  )
  return array_from_fig(fig)

def collate_timestep_img_attn(images):
  N = len(images)
  image_shape = images[0].shape

  # Create an empty array for the collated image
  collated_image = np.zeros((image_shape[0], image_shape[1]*N, image_shape[2]), dtype=np.uint8)

  # Collate the images into the single long image
  for i, image in enumerate(images):
      collated_image[:, i*image_shape[1]:(i+1)*image_shape[1], :] = image
  return collated_image

def slot_attn_entropy(attn, normalize: bool = True):
  # Assuming `attn` is your array of shape (T, N, M)
  # T = time
  # N = number of slots
  # M = number of spatial positions
  T, N, M = attn.shape

  # Step 1: Calculate entropy for each type at each time step
  entropy = -np.sum(attn * np.log2(attn), axis=-1)

  # Step 2: Normalize the entropy values
  if normalize:
    entropy = entropy / np.log2(M)

  # Step 3: Create the line plot
  time = np.arange(T)  # Time values for the x-axis

  fig, ax = plt.subplots()
  for i in range(N):
      ax.plot(time, entropy[:, i], label=f'Slot {i+1}')

  # Remove whitespace around the figure
  # fig.tight_layout()

  # Add labels and legend
  ax.set_xlabel('Time', fontsize=FONTSIZE)
  ax.set_ylabel('Normalized Entropy', fontsize=FONTSIZE)
  if normalize:
    ax.set_ylim(0, 1)
  ax.legend(fontsize=FONTSIZE)

  return array_from_fig(fig)

def slot_attn_max_likelihood(attn):
  # Assuming `attn` is your array of shape (T, N, M)
  # T = time
  # N = number of slots
  # M = number of spatial positions
  T, N, M = attn.shape

  # Step 3: Create the line plot
  time = np.arange(T)  # Time values for the x-axis

  fig, ax = plt.subplots()
  for i in range(N):
      ax.plot(time, attn[:, i].max(-1), label=f'Slot {i+1}')

  # Remove whitespace around the figure
  # fig.tight_layout()

  # Add labels and legend
  ax.set_xlabel('Time', fontsize=FONTSIZE)
  ax.set_ylabel('Max likelihood', fontsize=FONTSIZE)
  ax.set_ylim(0, 1)
  ax.legend(fontsize=FONTSIZE)

  return array_from_fig(fig)

###################
# data collection utils
###################

def default_collect_data(data: dict,
                         env,
                         actor,
                         timestep,
                         action,
                         tile_size: int=8):
  observation = timestep.observation.observation
  image = observation.image

  state = actor._state.recurrent_state
  attn = state.attn
  slots = attn.shape[0]
  # spatial = attn.shape[1]
  width = image.shape[1] // tile_size

  data['attn'].append(attn.reshape(slots, width, width))
  data['image'].append(image)

def asarry(data):
  for k, v in data.items():
      data[k] = np.asarray(v)
  return data

def collect_episode(env, actor,
                    data: dict=collections.defaultdict(list),
                    get_task_name= lambda env: "task",
                    collect_data = default_collect_data,
                    data_post_process=asarry):
    timestep = env.reset()
    actor.observe_first(timestep)

    task_name = get_task_name(env)
    while not timestep.last():
        # Generate an action from the agent's policy.

        action = actor.select_action(timestep.observation)
        collect_data(data=data,
                     env=env,
                     actor=actor,
                     timestep=timestep,
                     action=action)
        # Step the environment with the agent's selected action.
        timestep = env.step(action)
    data = data_post_process(data)
    return task_name, data
