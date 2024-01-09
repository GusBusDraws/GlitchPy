import math
import matplotlib.pyplot as plt
import numpy as np
import string

def image(
    img,
    **kwargs,
):
    nrows = 1
    ncols = 1
    fig, axes = plt.subplots(nrows, ncols, constrained_layout=True, **kwargs)
    if ncols == 1:
        axes = [axes]
    axes[0].imshow(img)
    axes[0].set_axis_off()
    return fig, axes

def images(
    imgs,
    vmin=None,
    vmax=None,
    imgs_per_row=None,
    fig_w=7.5,
    subplot_letters=False,
    dpi=100
):
    """Plot images.
    ----------
    Parameters
    ----------
    imgs : list
        List of NumPy arrays representing images to be plotted.
    imgs_per_row : int or None, optional
        Number of images to plot in each row. Default is None and all images
        are plotted in the same row.
    fig_w : float, optional
        Width of figure in inches, by default 7.5
    subplot_letters : bool, optional
        If true, subplot letters printed underneath each image.
        Defaults to False
    dpi : float, optional
        Resolution (dots per inch) of figure. Defaults to 300.
    -------
    Returns
    -------
    matplotlib.Figure, matplotlib.Axis
        2-tuple containing matplotlib figure and axes objects
    """
    # If single image passed, add it to a list
    if not isinstance(imgs, list):
        imgs = [imgs]
    # If single value passed for vmin or vmax, make a list full of that value
    if isinstance(vmin, int) or isinstance(vmin, float):
        vmin = [vmin] * len(imgs)
    if isinstance(vmax, int) or isinstance(vmax, float):
        vmax = [vmax] * len(imgs)
    if vmin == None:
        vmin = [None for _ in range(len(imgs))]
    if vmax == None:
        vmax = [None for _ in range(len(imgs))]
    n_imgs = len(imgs)
    img_w = imgs[0].shape[1]
    img_h = imgs[0].shape[0]
    if imgs_per_row is None:
        n_cols = n_imgs
    else:
        n_cols = imgs_per_row
    n_rows = int(math.ceil( n_imgs / n_cols ))
    fig_h = fig_w * (img_h / img_w) * (n_rows / n_cols)
    if subplot_letters:
        fig_h *= (1 + (0.12 * n_rows))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_w, fig_h), constrained_layout=True,
        dpi=dpi, facecolor='white'
    )
    if isinstance(axes, np.ndarray):
        ax = axes.ravel()
    else:
        # When only one image, wrap axis object into list to make iterable
        ax = [axes]
    for i, img in enumerate(imgs):
        ax[i].imshow(img, vmin=vmin[i], vmax=vmax[i], interpolation='nearest')
        if subplot_letters:
            letter = string.ascii_lowercase[i]
            ax[i].annotate(
                f'({letter})', xy=(0.5, -0.05),
                xycoords='axes fraction', ha='center', va='top', size=12)
    for a in ax:
        a.axis('off')
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

