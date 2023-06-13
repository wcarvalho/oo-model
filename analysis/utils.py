
from typing import List, Union
from absl import logging

import jax
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import wandb

import math


from acme import types as acme_types
from muzero import types as muzero_types


Array = acme_types.NestedArray
State = acme_types.NestedArray

FONTSIZE = 14

def array_from_fig(fig):
  # Save the figure to a numpy array
  fig.canvas.draw()
  image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  img = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close(fig)
  return img


def plot_image(image, title=""):
  # Create subplots
  fig, ax = plt.subplots()

  # Plot the image
  ax.imshow(image)

  # Add a title
  ax.set_title(title)

  # Remove ticks
  ax.set_xticks([])
  ax.set_yticks([])

  return array_from_fig(fig)

def plot_images(images, columns, width=5):

    C = columns
    N = len(images)
    R = math.ceil(N / C)

    fig, axes = plt.subplots(R, C, figsize=(width*C, width*R), facecolor='black')

    for i, ax in enumerate(axes.flat):
        if i < N:
            ax.imshow(images[i])
        ax.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    return array_from_fig(fig)

def compute_entropy(prob, mask=None, normalize=True):
  # Step 1: Calculate entropy for each type at each time step
  if mask is not None:
    if mask.ndim < prob.ndim:
      diff = prob.ndim - mask.ndim
      tile = prob.shape[:diff] + (1,)*(prob.ndim - diff)
      mask = np.tile(mask, tile)
    log_prob = np.where(mask, np.zeros_like(prob), np.log2(prob+1e-5))
  else:
    log_prob = np.log2(prob)
  entropy = -np.sum(prob * log_prob, axis=-1)

  # Step 2: Normalize the entropy values
  if normalize:
    M = prob.shape[-1]
    entropy = entropy / np.log2(M)

  return entropy


def plot_entropy(
    data: Union[List[acme_types.NestedArray], acme_types.NestedArray],
    labels: Union[List[str], str] = None,
    fontsize: int = None,
    xlabel: str = None,
    normalize: bool = True):
  if not isinstance(data, list):
    data = [data]
  if not isinstance(labels, list):
    labels = [labels]

  fontsize = fontsize or FONTSIZE

  # Step 3: Create the line plot
  N = len(data)
  fig, ax = plt.subplots()
  for i in range(N):
    entropy = data[i]
    T = len(entropy)
    x = np.arange(T)  # x values for the x-axis

    if labels:
      label = labels[i]
    else:
      label = f'Prob {i+1}'
    ax.plot(x, entropy, label=label)

  # Remove whitespace around the figure
  # fig.tight_layout()

  # Add labels and legend
  ax.set_xlabel(xlabel, fontsize=fontsize)
  ax.set_ylabel('Entropy', fontsize=fontsize)
  ax.grid()
  loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
  ax.xaxis.set_major_locator(loc)
  ymax = entropy.max()
  if normalize:
    ax.set_ylim(-.05, max(1.05, ymax*1.05))
  ax.legend(fontsize=fontsize)

  return array_from_fig(fig)

def plot_heatmap(
    data: Union[List[acme_types.NestedArray], acme_types.NestedArray],
    labels: Union[List[str], str] = None,
    fontsize: int = None):
  if not isinstance(data, list):
    data = [data]
  if not isinstance(labels, list):
    labels = [labels]

  num_plots = len(data)

  fig, axs = plt.subplots(num_plots, 1, figsize=(len(data[0]), num_plots), gridspec_kw={'hspace': 0})

  # Create a reversed colormap with white and black colors
  cmap = plt.cm.get_cmap('binary_r')

  for i in range(num_plots):
    datum = data[i]
    datum = datum.reshape(1, -1)
    if labels:
      label = labels[i]
    else:
      label = f'{i+1}'

    axs[i].imshow(datum, cmap=cmap, aspect='auto')
    axs[i].set_yticks(range(datum.shape[0]))
    axs[i].set_yticklabels([label], fontsize=fontsize)
    axs[i].tick_params(axis='both', which='both', length=0)
    axs[i].grid(axis='x', linestyle='--', linewidth=0.5, color='gray')
    loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
    axs[i].xaxis.set_major_locator(loc)

  axs[-1].set_xticks(np.arange(len(data[-1]) + 1) - 0.5)  # Set x-ticks in between the ticks
  return array_from_fig(fig)

def plot_line(
    ys: Union[List[acme_types.NestedArray], acme_types.NestedArray],
    xs: Union[List[acme_types.NestedArray], acme_types.NestedArray] = None,
    x_0: int = 0,
    labels: Union[List[str], str] = None,
    xlabel='', ylabel='', title=''):
  if not isinstance(ys, list):
    ys = [ys]
  if labels is not None and not isinstance(labels, list):
    labels = [labels]*len(ys)
  if xs is not None and not isinstance(xs, list):
    xs = [xs]*len(ys)

  fig, ax = plt.subplots()

  if labels is None:
    labels = [f"y_{idx+1}" for idx in np.arange(len(ys))]

  for idx in np.arange(len(ys)):
    if xs is None:
      x = np.arange(x_0, x_0+len(ys[idx]))  # Time values for the x-axis
    else:
      x = xs[0]
    ax.plot(x, ys[idx], label=labels[idx])

  ax.grid()
  loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
  ax.xaxis.set_major_locator(loc)
  ax.set_xlabel(xlabel, fontsize=FONTSIZE)
  ax.set_ylabel(ylabel, fontsize=FONTSIZE)
  ax.set_title(title, fontsize=FONTSIZE+2)
  ax.legend(fontsize=FONTSIZE)

  return array_from_fig(fig)

def plot_compare_pmfs(
    pmfs: List[np.ndarray],
    pmf_labels: List[str] = None,
    xlabels: List[str] = None,
    xlabel='',
    ylabel='',
    title='',
):
  num_pmfs = len(pmfs)
  bar_width = 0.8 / num_pmfs  # Adjust the bar width based on the number of pmfs


  # Set the positions of the bars on the x-axis
  pmf_labels = pmf_labels or [f'{idx+1}' for idx in range(pmfs[0].shape[0])]

  # Create the figure and axes
  fig, ax = plt.subplots()

  # Plot the bars for each pmf
  for i, pmf in enumerate(pmfs):
      x = np.arange(len(pmf))
      ax.bar(x + (i - len(pmfs)/2 + 0.5) * bar_width, pmf, width=bar_width, label=pmf_labels[i])

  # Set the labels, title, and legend
  ax.set_xlabel(xlabel, fontsize=12)
  ax.set_ylabel(ylabel, fontsize=12)
  ax.set_title(title, fontsize=14)
  ax.legend(fontsize=12)
  ax.set_xticks(x)
  ax.set_xticklabels(xlabels, fontsize=8)

  return array_from_fig(fig)
