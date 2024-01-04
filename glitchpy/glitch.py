import matplotlib.pyplot as plt
import numpy as np
import skimage


def convert_img(img, img_type, channels):
    if img_type != channels:
        if img_type == 'RGB':
            img = skimage.color.rgb2hsv(img)
            img = skimage.util.img_as_ubyte(img)
        elif img_type == 'HSV':
            img = skimage.color.hsv2rgb(img)
            img = skimage.util.img_as_ubyte(img)
        else:
            raise ValueError(f'img_type {img_type} not recognized')
    return img

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
        hist, bins = skimage.exposure.histogram(img[:, :, i])
        ax.plot(bins, hist, label=channels[i], color=f'C{i}')
        if n_multiotsu[i] != 0:
            thresh_vals = skimage.filters.threshold_multiotsu(img[:, :, i], n_multiotsu[i])
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
