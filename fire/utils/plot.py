import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

try:
    import rasterio as rio # for dataset reading
    import pyproj # for projection stuff
    from affine import Affine # class for transform matrices
    import cartopy.crs as ccrs # cartopy CRSs
except ModuleNotFoundError as e:
    warnings.warn("Exception caught during import: " + str(e), UserWarning)

import fire.utils.etc as uetc

from typing import Optional, List


def plot_onto_map(src: "rio.DatasetReader", crs: "ccrs", 
                  override_raster: Optional[np.ndarray] = None,
                  bands: List[int]=None, factor: float=None, tf: "Affine"=None,
                  figsize=(10,10), cmap=None
                 ) -> "cartopy.mpl.geoaxes.GeoAxesSubplot":
    """
    Quick and dirty plot function.
    Args:
        factor: hacky way to plot classification maps (e.g.) (low values)
        bands: 0-based! Works definitely with lists of length 3, e.g. [0,0,0].
        override_raster: overrides raster read from src, but still uses geo 
            properties from src. If not None, bands and factor is ignored. 
            Defaults to None.
    """
    if isinstance(src, rio.DatasetReader):
        # read image into ndarray
        im = src.read()
        
        # transpose the array from (band, row, col) to (row, col, band)
        im = np.transpose(im, [1,2,0])
    elif isinstance(src, np.ndarray): # format (row, col, band)
        im = src
    else:
        raise TypeError("src wrong type")
        
    if override_raster is not None:
        im = override_raster
    else:
        if bands:
            im = im[:,:,bands]
            print(im.shape)
        if factor:
            im *= factor
    
    # create figure
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': crs})
    ax.set_xmargin(0.05) # doesn't do anything???
    ax.set_ymargin(0.10) # doesn't do anything???
    
    # plot raster
    plt.imshow(im, origin='upper', 
               extent=[src.bounds.left, src.bounds.right, 
                       src.bounds.bottom, src.bounds.top], 
               transform=crs, interpolation='nearest', cmap=cmap)
    
    # plot coastlines
    ax.coastlines(resolution='10m', color='red', linewidth=1)
    
    return fig, ax


def add_jitter(x: np.ndarray, scale=None) -> np.ndarray:
    """
    Add random noise to an array-like.

    Args:
        scale (float): Passed to `scale` in `np.random.normal`.
    """
    noise = np.random.normal(size=x.shape, scale=scale)
    return x+noise

def rolling_mean(x, y, window_size: int, line_alpha: float=1.0, 
                 fill_alpha: float=.5, lw: float=1.5, color=None, ax=None, 
                 **plot_kwargs):
    if ax is None:
        raise NotImplementedError("pass an axis")

    order = np.argsort(x)
    x, y  = x[order], y[order]

    # mean line
    rmean_x = uetc.rolling_mean(x, window_size)
    rmean_y = uetc.rolling_mean(y, window_size)
    ax.plot(rmean_x, rmean_y, color=color, lw=lw, alpha=line_alpha, 
            **plot_kwargs)

    # std shade area
    rstd_y = uetc.rolling_std(y, window_size)
    ax.fill_between(rmean_x, rmean_y + rstd_y/2, rmean_y - rstd_y/2, 
                    alpha=fill_alpha, facecolor=color)
    
    return ax