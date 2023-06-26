import matplotlib.pyplot as plt
import numpy as np

def image(
    img,
    **kwargs,
):
    nrows = 1
    ncols = 1
    fig, axes = plt.subplots(nrows, ncols, **kwargs)
    if ncols == 1:
        axes = [axes]
    axes[0].imshow(img)
    axes[0].set_axis_off()
    return fig, axes

def channels(
    img,
    cmap='inferno',
    chan_range=[0, 255],
    **kwargs,
):
    nrows = 1
    ncols = img.shape[-1]
    fig, axes = plt.subplots(nrows, ncols, **kwargs)
    for i in range(ncols):
        axes[i].imshow(
            img[:, :, i], cmap=cmap,
            vmin=chan_range[0], vmax=chan_range[1]
        )
        axes[i].set_axis_off()
    return fig, axes

def histogram(img, nbins=256, channel_labels=None):
    if channel_labels == None:
        channel_labels = [0, 1, 2]
    fig, ax = plt.subplots()
    for chan in range(img.shape[-1]):
        hist, bins = np.histogram(img[..., chan], bins=nbins)
        ax.plot(bins[1:], hist, c=f'C{chan}', label=channel_labels[chan])
        ax.legend()
    return fig, ax

