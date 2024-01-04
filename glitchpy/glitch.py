import matplotlib.pyplot as plt
import numpy as np
from skimage import color, filters, exposure, util


def convert_img(img, img_type, channels):
    if img_type != channels:
        if img_type == 'RGB':
            img = color.rgb2hsv(img)
            img = util.img_as_ubyte(img)
        elif img_type == 'HSV':
            img = color.hsv2rgb(img)
            img = util.img_as_ubyte(img)
        else:
            raise ValueError(f'img_type {img_type} not recognized')
    return img

def game_of_life(grid, on=255, off=0):
    # copy grid since we require 8 neighbors
    # for calculation and we go line by line
    new_grid = np.zeros_like(grid)
    nrows = grid.shape[0]
    ncols = grid.shape[1]
    for i in range(nrows):
        for j in range(ncols):
            # Compute 8-neighbor sum using toroidal boundary
            # conditions: x and y wrap around so simulation takes
            # place on a toroidal surface (e.g. Pac-Man)
            # % is modulus; counts across rows as if rotary phone dial
            n_neighbors = (
                int(grid[(i-1) % nrows, (j-1) % ncols])
                + int(grid[(i-1) % nrows, j])
                + int(grid[(i-1) % nrows, (j+1) % ncols])
                + int(grid[i, (j-1) % ncols])
                + int(grid[i, (j+1) % ncols])
                + int(grid[(i+1) % nrows, (j-1) % ncols])
                + int(grid[(i+1) % nrows, j])
                + int(grid[(i+1) % nrows, (j+1) % ncols])
            ) // 255
            # Apply Conway's Game of Life rules
            if grid[i, j]  == on:
                # 1. Any live cell with fewer than two live neighbours dies,
                #    as if by underpopulation
                if n_neighbors < 2:
                    new_grid[i, j] = off
                # 2. Any live cell with two or three live neighbours lives on
                if (n_neighbors == 2) or (n_neighbors == 3):
                    new_grid[i, j] = on
                # 3. Any live cell with more than three live neighbours dies,
                #    as if by overpopulation
                if n_neighbors > 3:
                    new_grid[i, j] = off
            else:
                # 4. Any dead cell with exactly three live neighbours becomes
                #    a live cell, as if by reproduction
                if n_neighbors == 3:
                    new_grid[i, j] = on
    return new_grid

def plot_channel(
    img, 
    channel=None, 
    img_type='RGB',
    channels='HSV',
    cmap='gray', 
    **kwargs
):
    # Convert image if img_type doesn't match the channels to show
    img = convert_img(img, img_type, channels)
    fig, ax = plt.subplots(1, 1, **kwargs)
    if channel is not None:
        img = img[:, :, channel]
    ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    return fig, ax

def plot_hists(
    img, 
    channels=['hue', 'saturation', 'value'], 
    n_multiotsu=0,
     **kwargs
):
    n_chans = len(channels)
    # If n_multiotsu passed, must be a single value to apply to each channel, 
    # or a list of value to apply to each channel
    if n_multiotsu != 0:
        if isinstance(n_multiotsu, list) and len(n_multiotsu) != n_chans:
            raise ValueError(
                f'List of n_multiotsu must match number of channels ({n_chans})'
            )
    else:
        n_multiotsu = [n_multiotsu] * n_chans
    fig, ax = plt.subplots(**kwargs)
    for i in range(n_chans):
        hist, bins = exposure.histogram(img[:, :, i])
        ax.plot(bins, hist, label=channels[i], color=f'C{i}')
        if n_multiotsu[i] != 0:
            thresh_vals = filters.threshold_multiotsu(img[:, :, i], n_multiotsu[i])
            print(f'{channels[i].capitalize()}: {thresh_vals}')
            for val in thresh_vals:
                ax.axvline(val)
    ax.legend()
    return fig, ax

def plot_image(
    img, 
    img_type='RGB',
    channels='HSV',
    cmap='inferno',
    chan_range=[0, 255],
    **kwargs,
):
    nrows = 1
    ncols = 1
    if channels:
        ncols = len(channels) + 1
    fig, axes = plt.subplots(nrows, ncols, **kwargs)
    if ncols == 1:
        axes = [axes]
    axes[0].imshow(img)
    axes[0].set_axis_off()
    # Convert image if img_type doesn't match the channels to show
    img = convert_img(img, img_type, channels)
    for i in range(1, ncols):
        axes[i].imshow(
            img[:, :, i - 1], cmap=cmap, 
            vmin=chan_range[0], vmax=chan_range[1]
        )
        axes[i].set_title(channels[i - 1])
        axes[i].set_axis_off()
    return fig, axes

def plot_masks(
    img, 
    masks, 
    img_type='RGB',
    channels='HSV',
    **kwargs
):
    # Convert image if img_type doesn't match the channels to show
    img = convert_img(img, img_type, channels)
    nrows = 1
    ncols = len(masks) + 1
    if img_type.lower() == 'hsv':
        # Convert image to RGB to show
        img_rgb = convert_img(img, img_type, 'RGB')
    else:
        img_rgb = img
    fig, axes = plt.subplots(nrows, ncols, **kwargs)
    axes[0].imshow(img_rgb)
    axes[0].set_axis_off()
    for i, mask in enumerate(masks):
        axes[i + 1].imshow(mask, cmap='gray')
        axes[i + 1].set_axis_off()
    return fig, axes

def posterize_otsu(img, type='binary'):
    """Posterize image (replace colors with limited palette) by determining
    Otsu threshold of each channel and replacing value above and below the
    threshold.
    ----------
    Parameters
    ----------
    img : np.ndarray
        3-channel image to be posterized.
    type : str, optional
        Type of values to replace. Either 'binary' to replace values below
        threshold to 0 and above to 255, or 'max' to use the most frequently
        appearing value below/above the threshold. Defaults to 'binary'.
    """
    thresholds = [filters.threshold_otsu(img[..., chan]) for chan in range(3)]
    print('Thresholds:', thresholds)
    if type == 'binary':
        rgb_vals = [[0, 255] for c in range(3)]
    elif type == 'max':
        # Determine most frequently occurring values above and below threshold
        rgb_vals = [[] for c in range(3)]
        for chan in range(3):
            thresh = thresholds[chan]
            hist, bins = np.histogram(img[..., chan], bins=256)
            rgb_vals[chan].append(np.argmax(hist[:thresh]))
            rgb_vals[chan].append(thresh + np.argmax(hist[thresh: -1]))
    else:
        raise ValueError('type must be "binary" or "max"')
    print('RGB values:', rgb_vals)
    img_post = img.copy()
    print(img_post.shape)
    for chan in range(3):
        img_post[img[..., chan] < thresholds[chan], chan] = rgb_vals[chan][0]
        img_post[img[..., chan] >= thresholds[chan], chan] = rgb_vals[chan][1]
    print(img_post.shape)
    return img_post

