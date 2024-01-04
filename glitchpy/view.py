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

def histogram(
    img,
    nbins=256,
    channel_thresholds=None,
    channel_labels=None,
    return_hist=False
):
    colors = ['C3', 'C2', 'C0']
    if channel_labels == None:
        channel_labels = [0, 1, 2]
    # Compile single values into nested list by channel
    if channel_thresholds is not None:
        for i in range(3):
            if not isinstance(channel_thresholds[i], list):
                channel_thresholds[i] = [channel_thresholds[i]]
    fig, ax = plt.subplots()
    hist_list = []
    bins_list = []
    for chan in range(img.shape[-1]):
        hist, bins = np.histogram(img[..., chan], bins=nbins)
        hist_list.append(hist)
        bins_list.append(bins)
        ax.plot(bins[1:], hist, c=colors[chan], label=channel_labels[chan])
        if channel_thresholds is not None:
            for thresh in channel_thresholds[chan]:
                ax.axvline(thresh, c=colors[chan])
        ax.legend()
    if return_hist:
        return hist_list, bins_list
    else:
        return fig, ax

