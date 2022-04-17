import matplotlib.pyplot as plt
import numpy as np
import skimage


def plot_channels(img, cmap='inferno', **kwargs):
    fig, axes = plt.subplots(1, 3, **kwargs)
    for chan_i, chan in enumerate(['hue', 'saturation', 'value']):
        ax = axes[chan_i]
        ax.imshow(img[:, :, chan_i], cmap=cmap)
        ax.set_title(chan)
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
