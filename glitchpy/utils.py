import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import color, filters, exposure, util
# Local imports
# import glitch

def get_colors_by_count(img, ncolors='all'):
    if ncolors != 'all' and isinstance(ncolors, int):
        raise ValueError('If not "all", ncolors must be an integer')
    # This function is based on the following answer:
    # https://stackoverflow.com/a/30901841/11395993
    # Lexicographically sort
    sorted_arr = img[np.lexsort(img.T), :]
    # Get the indices where a new row appears
    diff_idx = np.where(np.any(np.diff(sorted_arr, axis=0), 1))[0]
    # Get the unique rows
    unique_rows = [sorted_arr[i] for i in diff_idx] + [sorted_arr[-1]]
    # Get the number of occurences of each unique array (the -1 is needed at
    # the beginning, rather than 0, because of fencepost concerns)
    counts = np.diff(
        np.append(np.insert(diff_idx, 0, -1), sorted_arr.shape[0] - 1))
    # Return the (row, count) pairs sorted by count
    colors_by_count = sorted(
        zip(unique_rows, counts), key=lambda x: x[1], reverse=True)
    if ncolors == 'all':
        return colors_by_count
    else:
        return colors_by_count[:ncolors]

def get_df_from_masks(img, masks, color_downscale=10):
    if color_downscale is not None:
        # Reduce colors in base image by rounding
        img = reduce_color_by_rounding(img, color_downscale)
    # Create dataframe to hold palette info
    df = pd.DataFrame(
        columns=['mask_i', 'red', 'green', 'blue', 'counts', 'grey_dist'])
    # Add common colors within each masked region
    for mask_i, mask in enumerate(masks):
        # Count colors in masked image
        colors_by_count = get_colors_by_count(img[mask])
        common_colors, counts = zip(*colors_by_count)
        common_colors = np.array(common_colors)
        # Add colors to dataframe
        df_i = pd.DataFrame(
            data=common_colors, columns=['red', 'green', 'blue'])
        df_i['mask_i'] = [mask_i] * len(counts)
        df_i['counts'] = counts
        df_i['grey_dist'] = (
            abs(df_i.red - df_i.green)
            + abs(df_i.green - df_i.blue)
            + abs(df_i.blue - df_i.red)
        )
        df = pd.concat([df, df_i])
    return df

def get_palette_from_df(df, min_grey_dist=100):
# def get_palette(df, ncolors, color_downscale=10, min_grey_dist=100):
    # df = get_palette_df(img, ncolors, color_downscale=color_downscale)
    # Get the most common color within each mask
    nmasks = df.mask_i.max() + 1
    palette = []
    for m_i in range(nmasks):
        most_common_color = df.loc[
            (df.mask_i == m_i)
            & (df.grey_dist >= min_grey_dist),
        ][['red', 'green', 'blue']].values[0].astype(int)
        palette.append(most_common_color)
    return palette

def get_palette_df(
        img, ncolors, color_downscale=10, also_return_percentile_masks=False):
    # Get hue image
    img_hsv = color.rgb2hsv(img)
    img_hue = img_hsv.copy()
    img_hue[..., 1] = 1
    img_hue[..., 2] = 1
    img_hue = color.hsv2rgb(img_hue)
    img_hue = util.img_as_ubyte(img_hue)
    img_hue_hsv = color.rgb2hsv(img_hue)
    # Split hue channel by percentile
    p = [i * 100 / ncolors for i in range(1, ncolors + 1)]
    thresholds = np.percentile(img_hue_hsv[..., 0], p)
    # Segment image according to percentile threshold values
    img_semantic = isolate_classes(
        img_hue_hsv[..., 0], threshold_values=thresholds)
    percentile_masks = []
    for i in range(ncolors):
        mask = img_semantic == i
        percentile_masks.append(mask)
    if color_downscale is not None:
        # Reduce colors in base image by rounding
        img = reduce_color_by_rounding(img, color_downscale)
    # Create dataframe to hold palette info
    df = pd.DataFrame(
        columns=['mask_i', 'red', 'green', 'blue', 'counts', 'grey_dist'])
    # Add common colors within each masked region
    for mask_i, mask in enumerate(percentile_masks):
        # Count colors in masked image
        colors_by_count = get_colors_by_count(img[mask])
        common_colors, counts = zip(*colors_by_count)
        common_colors = np.array(common_colors)
        # Add colors to dataframe
        df_i = pd.DataFrame(data=common_colors, columns=['red', 'green', 'blue'])
        df_i['mask_i'] = [mask_i] * len(counts)
        df_i['counts'] = counts
        df_i['grey_dist'] = (
            abs(df_i.red - df_i.green)
            + abs(df_i.green - df_i.blue)
            + abs(df_i.blue - df_i.red)
        )
        df = pd.concat([df, df_i])
    if not also_return_percentile_masks:
        return df
    else:
        return df, percentile_masks

def isolate_classes(
    img,
    threshold_values,
    start=1,
    intensity_step=1,
):
    """Threshold array with multiple threshold values to separate classes
    (semantic segmentation).
    ----------
    Parameters
    ----------
    img : numpy.ndarray
        MxNxD array (D slices, M rows, N columns) NumPy array to be segmented
        according to threshold values.
    threshold_values : numpy.ndarray or list of lists
        DxT array where D matches the MxNxD array img containing values to
        segment image.
    intensity_step : int, optional
        Step value separating intensities. Defaults to 1, but might be set to
        soemthing like 125 such that isolated classes could be viewable in
        saved images.
    -------
    Returns
    -------
    numpy.ndarray
        MxNxD array representing semantic segmentation.
    """
    if len(img) == 3:
        nchans = img.shape[2]
    else:
        nchans = 1
    if not isinstance(threshold_values, (list, np.ndarray)):
        threshold_values = [threshold_values]
    # Sort thresh_vals in ascending order then reverse to get largest first
    threshold_values.sort()
    img_semantic = np.zeros_like(img, dtype=np.uint8)
    # Starting with the lowest threshold value, set pixels above each
    # increasing threshold value to an increasing unique marker (1, 2, etc.)
    # multiplied by the intesnity_step parameter
    i = 0
    if nchans > 1:
        for c in range(nchans):
            for val in threshold_values[c]:
                img_semantic[img[..., c] > val, c] = int(
                    (start + i) * intensity_step)
                i += 1
    else:
            for val in threshold_values:
                img_semantic[img > val] = int((start + i) * intensity_step)
                i += 1
    return img_semantic

def reduce_color_by_rounding(img, scale=5):
    img_reduced = np.round(img / scale).astype(int) * scale
    return img_reduced

def save_images(
    imgs,
    save_dir,
    file_suffix='png',
    img_names=None,
    overwrite=False,
):
    """Save images to save_dir.
    ----------
    Parameters
    ----------
    imgs : numpy.ndarray or list
        Images to save, either as a list or a 3D numpy array (4D array of
        colored images also works)
    save_dir : str or Path
        Path to new directory to which iamges will be saved. Directory must
        not already exist to avoid accidental overwriting.
    img_names : list, optional
        List of strings to be used as image filenames when saved. If not
        included, images will be names by index. Defaults to None.
    overwrite : bool, optional
        If True, existing directory will be overwritten. Defaults to False.
    """
    save_dir = Path(save_dir)
    file_suffix = file_suffix.lstrip('.')
    # Create directory and raise error if dir already exists and overwrite False
    if not overwrite:
        save_dir.mkdir(parents=True, exist_ok=False)
    else:
        save_dir.mkdir(parents=True, exist_ok=True)
    # If imgs is a numpy array and not a list, convert it to a list of images
    if isinstance(imgs, np.ndarray):
        # If 3D: (slice, row, col)
        if len(imgs.shape) == 3:
            imgs = [imgs[i, :, :] for i in range(imgs.shape[0])]
        # If 4D: (slice, row, col, channel) where channel is RGB (color) value
        elif len(imgs.shape) == 4:
            imgs = [
                util.img_as_ubyte(imgs[i, :, :, :])
                for i in range(imgs.shape[0])
            ]
    for i, img in enumerate(imgs):
        # if no img_names, use the index of the image
        if img_names is None:
            n_imgs = len(imgs)
            # Pad front of image name with zeros to match longest number
            img_name = str(i).zfill(len(str(n_imgs)))
        else:
            img_name = img_names[i]
        iio.imwrite(Path(save_dir / f'{img_name}.{file_suffix}'), img)
    print(f'{len(imgs)} image(s) saved to: {save_dir.resolve()}')

def split_semantic(img_semantic):
    nmasks = img_semantic.max() + 1
    masks = []
    for mask_i in range(nmasks):
        mask = img_semantic == mask_i
        masks.append(mask)
    return masks

def threshold_multi_otsu(
    imgs,
    nclasses=2,
    return_fig_ax=False,
    ylims=None,
    **kwargs
):
    """Semantic segmentation by application of the Multi Otsu algorithm.
    ----------
    Parameters
    ----------
    imgs : numpy.ndarray
        3D NumPy array representing images for which thresholds will be
        determined.
    nclasses : int
        Number of classes to  used to calculate histogram.
    -------
    Returns
    -------
    list
        List of intensity minima that can be used to threshold the image.
        Values will be 16-bit if imgs passed is 16-bit, else float.
    """
    print('Calculating Multi Otsu thresholds...')
    imgs_flat = imgs.flatten()
    thresh_vals = filters.threshold_multiotsu(imgs_flat, nclasses)
    # Calculate histogram
    hist, hist_centers = exposure.histogram(imgs, nbins=256)
    if return_fig_ax:
        # Plot peaks & mins on histograms
        fig, ax = plt.subplots()
        ax.plot(hist_centers, hist, label='Histogram')
        if ylims is not None:
            ax.set_ylim(ylims)
        ymin, ymax = ax.get_ylim()
        ax.vlines(
            x=thresh_vals, ymin=ymin, ymax=ymax, colors='C2',
            label='Thresholds'
        )
        ax.legend()
        return thresh_vals, fig, ax
    else:
        return thresh_vals

