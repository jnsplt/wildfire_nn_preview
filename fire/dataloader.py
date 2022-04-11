#
import numpy as np
import pandas as pd
from datetime import datetime
import os
import warnings
from glob import glob
import time
import calendar
from tqdm import tqdm

from functools import wraps # required for decorator FeatureLoader._lazy

from typing import List, Tuple, Optional, Callable, Union, Iterable, Mapping, \
                   Any, Dict

# geo stuff
try:
    import rasterio as rio # for dataset reading
    from rasterio.errors import RasterioIOError
    import pyproj # for projection stuff
    from affine import Affine # class for transform matrices
    import xarray as xr

    import fire.utils.geo as ugeo
except ModuleNotFoundError as e:
    warnings.warn("Exception caught during import: " + str(e), UserWarning)

# own stuff
import fire.utils.parallel as upar
import fire.utils.modis as umod
import fire.utils.io as uio
import fire.utils.etc as uetc
import fire.utils.pandas as upd
import fire.features as feat


def get_nonfire_wilderness(mod14a1_files: List[str], mcd12q1_files: List[str], 
                           offsets="center", dst_crs=None, verbose: bool = True, 
                           min_corresponding_mcd_files: int = 2, **kwargs
                          ) -> pd.DataFrame: 
    """
    Args:
        **kwargs: passed to `where()`. `max_coords` makes sense here.
    """
    all_dfs = []

    condition = lambda fire_raster, is_wilderness_raster: \
        (fire_raster < 7) & is_wilderness_raster
    
    if verbose:
        progress = uetc.ProgressDisplay(len(mod14a1_files))
        progress.start_timer().print_status()

    for mod14a1 in mod14a1_files:
        mcds = umod.get_corresponding_mcd12q1_files(mod14a1, mcd12q1_files)
        mcds = [f for f in mcds if f is not None]
        if len(mcds) >= min_corresponding_mcd_files:
            all_dfs.append(_get_where_wilderness(
                mod14a1, mcds, condition, **kwargs))
        if verbose:
            progress.update_and_print()

    if verbose:
        progress.stop()
    
    #todo: should not fail if there's no df in the list
    return pd.concat(all_dfs, axis=0)


def get_nonfire_wild_veg(mod14a1_files: List[str], mcd12q1_files: List[str], 
                         offsets="center", dst_crs=None, verbose: bool = True, 
                         min_corresponding_mcd_files: int = 2, **kwargs
                        ) -> pd.DataFrame: 
    """
    Args:
        **kwargs: passed to `where()`. `max_coords` makes sense here.
    """
    all_dfs = []

    condition = lambda fire_raster, wild_veg_raster: \
        (fire_raster < 7) & wild_veg_raster
    
    if verbose:
        progress = uetc.ProgressDisplay(len(mod14a1_files))
        progress.start_timer().print_status()

    for mod14a1 in mod14a1_files:
        mcds = umod.get_corresponding_mcd12q1_files(mod14a1, mcd12q1_files)
        mcds = [f for f in mcds if f is not None]
        if len(mcds) >= min_corresponding_mcd_files:
            all_dfs.append(_get_where_wild_vegetation(
                mod14a1, mcds, condition, **kwargs))
        if verbose:
            progress.update_and_print()

    if verbose:
        progress.stop()
    
    #todo: should not fail if there's no df in the list
    return pd.concat(all_dfs, axis=0)


# def _get_wildfires_from_single_file(mod14a1_file: str, 
#                                     mcd12q1_files: Iterable[str], 
#                                     offsets="center", dst_crs=None
#                                    ) -> pd.DataFrame:
#     """
#     Wrapper for `where()` and MOD14A1 fire pixels, filtered by "wilderness" land 
#     cover types in MCD12Q1, i.e. NOT 12, 13, or 14.

#     Args:
#         mod14a1_file (str): MOD14A1.006 HDF file paths. 
#         mcd12q1_files (tuple or list of str): MCD12Q1 files from which the 
#             tiles at the fire locations (all) have to be wilderness. 
#         dst_crs, offsets: Passed on to `where()`.
    
#     Returns:
#         pandas.DataFrame: see `where()`.
#     """
#     condition = lambda fire_raster, is_wilderness_raster: \
#         (fire_raster >= 7) & is_wilderness_raster

#     # wilderness raster
#     is_wilderness_raster = get_wilderness_raster(mcd12q1_files)
#     is_wilderness_raster = uetc.pool(is_wilderness_raster, (2,2), "min")

#     firemask_sds_index = 0
#     df = _get_wrapper(files=[mod14a1_file], condition=condition, 
#                       sds_index=firemask_sds_index, offsets=offsets, 
#                       dst_crs=dst_crs, incl_vh=True, 
#                       cond_args=[is_wilderness_raster], verbose=False)
#     return df.rename(columns={"val": "fire_val"})


# def hacky_random_nonfire_wilderness(fire_raster, is_wilderness_raster):
#     x = (fire_raster < 7) & is_wilderness_raster
#     n = 130
#     choices = [True]*n+[False]*(1200**2-n)
#     m = np.random.choice(choices, size=fire_raster.shape, replace=False)
#     x = x*m
#     return x


def get_wildfires(mod14a1_files: List[str], mcd12q1_files: List[str], 
                  offsets="center", dst_crs=None, verbose: bool = True, 
                  min_corresponding_mcd_files: int = 2
                 ) -> pd.DataFrame: 
    all_dfs = []

    condition = lambda fire_raster, wild_veg_raster: \
        (fire_raster >= 7) & wild_veg_raster
    
    if verbose:
        progress = uetc.ProgressDisplay(len(mod14a1_files))
        progress.start_timer().print_status()

    for mod14a1 in mod14a1_files:
        mcds = umod.get_corresponding_mcd12q1_files(mod14a1, mcd12q1_files)
        mcds = [f for f in mcds if f is not None]
        if len(mcds) >= min_corresponding_mcd_files:
            all_dfs.append(_get_where_wild_vegetation(mod14a1, mcds, condition))
        if verbose:
            progress.update_and_print()

    if verbose:
        progress.stop()
    
    #todo: should not fail if there's no df in the list
    return pd.concat(all_dfs, axis=0).astype({"row": int, "col": int})


def _get_where_wilderness(mod14a1_file: str, 
                          mcd12q1_files: Iterable[str],
                          condition: Callable,
                          offsets="center", dst_crs=None, **kwargs
                         ) -> pd.DataFrame:
    """
    Wrapper for `where()` and MOD14A1 fire pixels, filtered by "wilderness" land 
    cover types in MCD12Q1, i.e. NOT 12, 13, or 14.

    Args:
        mod14a1_file (str): MOD14A1.006 HDF file paths. 
        mcd12q1_files (tuple or list of str): MCD12Q1 files from which the 
            tiles at the fire locations (all) have to be wilderness. 
        condition (callable): Function with signature 
            `f(fire_raster, is_wilderness_raster)`.
        dst_crs, offsets: Passed on to `where()`.
    
    Returns:
        pandas.DataFrame: see `where()`.
    """
    # wilderness raster
    is_wilderness_raster = get_wilderness_raster(mcd12q1_files)
    is_wilderness_raster = uetc.pool(is_wilderness_raster, (2,2), "min")

    firemask_sds_index = 0
    df = _get_wrapper(files=[mod14a1_file], condition=condition, 
                      sds_index=firemask_sds_index, offsets=offsets, 
                      dst_crs=dst_crs, incl_vh=True, 
                      cond_args=[is_wilderness_raster], verbose=False, 
                      **kwargs)
    return df.rename(columns={"val": "fire_val"})


def _get_where_wild_vegetation(mod14a1_file: str, 
                               mcd12q1_files: Iterable[str],
                               condition: Callable,
                               offsets="center", dst_crs=None, **kwargs
                               ) -> pd.DataFrame:
    """
    Wrapper for `where()` and MOD14A1 fire pixels, filtered by "wilderness" land 
    cover types in MCD12Q1 which are vegetated, i.e. 1-11.

    Args:
        mod14a1_file (str): MOD14A1.006 HDF file paths. 
        mcd12q1_files (tuple or list of str): MCD12Q1 files from which the 
            tiles at the fire locations (all) have to be wilderness. 
        condition (callable): Function with signature 
            `f(fire_raster, wild_veg_raster)`.
        dst_crs, offsets: Passed on to `where()`.
    
    Returns:
        pandas.DataFrame: see `where()`.
    """
    # wilderness raster
    wild_veg_raster = get_wild_vegetation_raster(mcd12q1_files)
    wild_veg_raster = uetc.pool(wild_veg_raster, (2,2), "min")

    firemask_sds_index = 0
    df = _get_wrapper(files=[mod14a1_file], condition=condition, 
                      sds_index=firemask_sds_index, offsets=offsets, 
                      dst_crs=dst_crs, incl_vh=True, 
                      cond_args=[wild_veg_raster], verbose=False, 
                      **kwargs)
    return df.rename(columns={"val": "fire_val"})


def get_wild_vegetation_raster(mcd12q1_files: Union[str,Iterable[str]]
                              ) -> np.ndarray:
    """
    all files have to be for the same vh tile; is not asserted!
    """
    if type(mcd12q1_files) is str:
        mcd12q1_files = [mcd12q1_files]

    wild_veg_lcts = list(range(1,12)) # land cover types for wild vegetation
    output_raster = None
    for f in mcd12q1_files:
        rio_sds    = rio.open(uio.get_subdataset_path(f, 0))
        lct_raster = rio_sds.read(1) # annual IGBP land cover type classifications

        wild_veg_raster = np.isin(lct_raster, wild_veg_lcts) # bool raster
        if output_raster is None:
            output_raster = wild_veg_raster
        else:
            output_raster &= wild_veg_raster

    return output_raster


def get_wilderness_raster(mcd12q1_files: Union[str,Iterable[str]]) -> np.ndarray:
    """
    all files have to be for the same vh tile; is not asserted!
    """
    if type(mcd12q1_files) is str:
        mcd12q1_files = [mcd12q1_files]

    human_lcts    = [12,13,14] # land cover types that have human activity
    output_raster = None
    for f in mcd12q1_files:
        rio_sds    = rio.open(uio.get_subdataset_path(f, 0))
        lct_raster = rio_sds.read(1) # annual IGBP land cover type classifications

        is_wilderness_raster = ~np.isin(lct_raster, human_lcts) # bool raster
        if output_raster is None:
            output_raster = is_wilderness_raster
        else:
            output_raster &= is_wilderness_raster

    return output_raster


def get_fires(files: List[str], offsets="center", dst_crs=None) -> pd.DataFrame:
    """
    Wrapper for `where()` and MOD14A1 fire pixels. 

    Args:
        files (list of str): MOD14A1.006 HDF file paths. 
        dst_crs, offsets: Passed on to `where()`.
    
    Returns:
        pandas.DataFrame: see `where()`.
    """
    condition          = lambda raster: raster >= 7
    firemask_sds_index = 0
    df = _get_wrapper(files=files, condition=condition, 
                      sds_index=firemask_sds_index, offsets=offsets, 
                      dst_crs=dst_crs, incl_vh=True)
    return df.rename(columns={"val": "fire_val"})


def get_human_activity(files: List[str], offsets="center", dst_crs=None
                      ) -> pd.DataFrame:
    """
    Wrapper for `where()` and MCD12Q1 human activity classes. 

    Args:
        files (list of str): MCD12Q1.006 HDF file paths. 
        dst_crs, offsets: Passed on to `where()`.
    
    Returns:
        pandas.DataFrame: see `where()`.
    """
    condition      = lambda raster: np.isin(raster, [12,13,14])
    igbp_sds_index = 0
    df = _get_wrapper(files=files, condition=condition, sds_index=igbp_sds_index,
                      offsets=offsets, dst_crs=dst_crs, incl_vh=True)
    return df.rename(columns={"val": "land_cover_type"})


def _get_wrapper(files: List[str], condition: Callable[[np.ndarray], np.ndarray], 
                 sds_index: int, offsets="center", dst_crs=None, 
                 incl_vh: bool=False, cond_args: Iterable[Any] = (), 
                 cond_kwargs: Mapping[str, Any] = {},
                 verbose: bool = True, **kwargs
                ) -> pd.DataFrame:
    """
    Args:
        **kwargs: passed to `where()`.
    """
    all_dfs = list()

    if verbose:
        progress = uetc.ProgressDisplay(len(files))
        progress.start_timer().print_status()
    for f in files:
        try:
            sds_path = uio.get_subdataset_path(f, 0)
            where_df = where(sds_path, condition=condition, 
                             offsets=offsets, dst_crs=dst_crs,
                             cond_args=cond_args, cond_kwargs=cond_kwargs,
                             **kwargs)
            if incl_vh:
                v, h = umod.v_from_hdf_filename(f), umod.h_from_hdf_filename(f)
                where_df.loc[:, "v"] = v if where_df.shape[0] > 0 else []
                where_df.loc[:, "h"] = h if where_df.shape[0] > 0 else []
            all_dfs.append(where_df)
        except RasterioIOError:
            print(f"\nFile {f} could not be read.\n")
        if verbose:
            progress.update_and_print()

    if verbose:
        progress.stop()
    return pd.concat(all_dfs, axis=0).reset_index(drop=True)


#todo not only date but time as well?
#todo naming consistent with numpy(.where)? with pandas(.query)?
#todo include i,j in output
def where(sds: str, condition: Callable[[np.ndarray], np.ndarray],
          offsets: Union[List[str], str] = "center", 
          dst_crs: Optional["pyproj.crs.CRS"] = None,
          cond_args: Iterable[Any] = (), cond_kwargs: Mapping[str, Any] = {},
          incl_rowcol: bool = True, max_coords: Optional[int] = None
         ) -> pd.DataFrame:
    """
    Args:
        sds (str): Path to a MODIS science dataset.
        condition (callable): Function that takes a numpy array of shape 
            (n_rows, n_cols) as input and outputs a numpy array of the same 
            shape but with boolean values only. See examples below.
        offsets (list of str, or str): Coordinates are computed for each offset.
            Values must be one of: center, ul, ur, ll, lr.
        dst_crs (pyproj.CRS (opt)): CRS to state coordinates in. If None,  
            EPSG:4326 (lat lon) will be used, and coordinate columns in output 
            dataframe will be named `lon` and `lat` instead of `x` and `y`. 
            Defaults to None.
        cond_args: Positional arguments to pass to `condition` in addition to 
            the raster. 
        cond_kwargs: Additional keyword arguments to pass as keyword arguments 
            to `condition`. 
        incl_rowcol (bool): If True, the row and column indices of the raster 
            where the condition is met are returned as two more columns in the
            output dataframe.
        max_coords (int (opt)): Maximum number of coordinates per vh-tile. If 
            the number of non-fire-wilderness coordinates exceed `max_coords`, 
            a subset of coordinates is chosen randomly.
    
    Returns: 
        pandas.DataFrame: with columns:
            "y" or "lat": y- or lat-coordinate of a pixel that met the 
                condition. 
            "x" or "lon": x- or lon-coordinate of a pixel that met the 
                condition. 
            "val": Value of the respective pixel in the raster.
            "date": Date of the respective raster. See details below.
            "offset": Offset for which x and y (or lon and lat) were calculated. 
                One of: center, ul, ur, ll, lr. 
            "row": See arg `incl_rowcol`.
            "col": See arg `incl_rowcol`.

    Details:
        Output column "date":
            If the science data set has only one date (or represents a range, 
            e.g. a year like in MCD12Q1 files), the date returned in the date 
            column is the beginning date.

    Example:
        # get all lat lon coordinates of fires in a MOD14A1 hdf file.
        # 0) imports
        import fire.utils.io as uio
        import fire.dataloader as fdl

        # 1) get science dataset path from HDF file
        #    You have to know the index for example from the MODIS doc
        index_of_firemask_sds = 0
        firemask_sds_path     = uio.get_subdataset_path(
            "MOD14A1.A2011361.h08v05.006.2015229220047.hdf", 
            index_of_firemask_sds)

        # 2) Define the condition. Since we want to extract all pixel locations
        #    of fires, we have to filter the raster data for  values >= 7
        #    (see MOD14A1 doc)
        fire_condition = lambda raster: raster >= 7

        # or
        def fire_condition(raster):
            return raster >= 7

        # 3) compute
        all_fires_in_lat_lon = fdl.where(firemask_sds_path, fire_condition)
    """
    # preprocess input args
    if type(offsets) is str:
        offsets = [offsets]

    # open sub/scientific-dataset
    rio_sds = rio.open(sds, mode="r")

    # get dates available in subdataset
    #todo outsource to own function if other hdf files have even more different
    # date namings
    dates = rio_sds.get_tag_item("Dates")
    if  dates is not None:
        dates = dates.split() # e.g. for MOD14A1
    else:
        dates = [rio_sds.get_tag_item("RANGEBEGINNINGDATE")] # e.g. for MCD12Q1
    dates = [datetime.strptime(d, r"%Y-%m-%d") for d in dates]

    # iterate over dates
    all_dfs = list() # will hold one dataframe for each date
    for i, d in enumerate(dates):
        raster_of_date_i = rio_sds.read()[i]
        is_true          = condition(raster_of_date_i, *cond_args, **cond_kwargs)

        if np.any(is_true):
            # pixel row- and col-ids of values that matched the condition (True)
            ii, jj       = np.where(is_true)
            if max_coords:
                random_selection = np.random.choice(
                    np.arange(len(ii)), size=max_coords, replace=False)
                ii, jj = ii[random_selection], jj[random_selection]

            pixel_values = raster_of_date_i[ii, jj]

            for o in offsets:
                xs, ys = ugeo.get_coords_for_pixels(
                    rio_sds, rows = ii, cols = jj, 
                    offset = o, dst_crs = dst_crs)

                df = pd.DataFrame({
                    "y": ys, 
                    "x": xs, 
                    "val": pixel_values, 
                    "date": d,
                    "offset": o
                })

                if incl_rowcol:
                    df.loc[:,"row"] = ii
                    df.loc[:,"col"] = jj

                all_dfs.append(df)


    # close sub/scientific-dataset
    rio_sds.close()

    if len(all_dfs) == 0:
        output_df = pd.DataFrame({
            "y": [], "x": [], "val": [], "date": [], "offset": []})
        if incl_rowcol:
            output_df.loc[:,"row"] = []
            output_df.loc[:,"col"] = []
    else:
        output_df = pd.concat(all_dfs, axis=0)

    if dst_crs is None:
        # rename x and y columns to lon and lat
        return output_df.rename(columns={"x": "lon", "y": "lat"})

def make_lct_index(v: int, h: int, year: int, mcd12q1_fpaths: Iterable[str], 
                   drop_lcts: set={13,15,16,255}, subset_ratio: float=1.0,
                   margin: int=0, return_ratios: bool=False, rnd_state=None
                  ) -> Union[pd.DataFrame, Tuple[pd.DataFrame,Dict[str,float]]]:
    """
    Creates a dataframe of Land Cover Types (LCTs) from MODIS MCD12Q1 files. 

    Requires two MCD12Q1 files: one for the year passed to this function and 
    the subsequent one. Only LCT classifications are kept / returned in the 
    dataframe, which do not change for a given pixel from year `year` to year 
    `year+1`.

    Parameters
    ----------
    v : int
        [description]
    h : int
        [description]
    year : int
        [description]
    mcd12q1_fpaths : Iterable[str]
        [description]
    drop_lcts : set, optional
        [description], by default {13,15,16,255}. Defaults refer to
        13 (urban/built-up), 15 (perm. snow/ice), 16 (barren), and 
        255 (unclassified).
    subset_ratio : float, optional
        Ratio of pixels to consider, by default 1.0. If < 1, pixels are dropped
        randomly (e.g. in order to reduce memory usage).
    margin : int, optional
        See details (3), by default 100. The default value of 100 refers to 
        50 km (MCD12Q1 has a spatial resolution of 500 m per pixel).
    return_ratios : bool, optional
        If True, a dictionary with True-ratios for each mask is returned as 
        second value, by default False.
    rnd_state : [type], optional
        Random state or seed to use for `subset_ratio`, by default None.

    Returns
    -------
    lct_index : pd.DataFrame
        columns:
            year (uint16): as passed
            v    (uint8): as passed
            h    (uint8): as passed
            i    (uint16): row of pixel
            j    (uint16): column of pixel
            lct  (uint8): extracted from MCD12Q1 layer 'LC_Type1' (IGBP)
    
    < mask_ratios > : Dict[str,float] : returned iff `return_ratios=True`
        [description]

    Details
    -------
    The requirements for a grid-cell / pixel to be recorded in the dataframe 
    are as follows (layers refer to MCD12Q1):
        1) Layer `QC` (Quality Assurance): pixel has to be 0 (classified land)
        2) Layer `LC_Type1` (IGBP):
            a) LCT value is NOT in `drop_lcts`
            b) LCT value doesn't change from year `year` to year `year+1`
        3) the distance from the pixel to the edge of the vh-tile (the LCT 
           raster at hand) amounts at least to `margin` pixels.

    [MCD12Q1] https://lpdaac.usgs.gov/documents/101/MCD12_User_Guide_V6.pdf
    """
    overall_mask, ratios, lct0 = _make_mcd12q1_mask(
        v=v, h=h, year=year, mcd12q1_fpaths=mcd12q1_fpaths, 
        drop_lcts=drop_lcts, subset_ratio=subset_ratio,
        margin=margin, return_ratios=return_ratios, rnd_state=rnd_state
    )
    
    # get row and column indices of True values in overall_mask
    ii, jj = np.nonzero(overall_mask) # row and col indices of okay pixels

    # extract lct values
    lcts = lct0[ii, jj]

    # make dataframe
    lct_index = (
        pd.DataFrame({"year": year, "v": v, "h": h, 
                      "i2": ii, "j2": jj, "lct": lcts})
        # reduce memory usage
        .astype({"year": "uint16", "v": "uint8", "h": "uint8", 
                 "i2": "uint16", "j2": "uint16", "lct": "uint8"})
    )
    
    if return_ratios:
        return lct_index, ratios
    else:
        return lct_index

def _make_mcd12q1_mask(v: int, h: int, year: int, mcd12q1_fpaths: Iterable[str], 
                       drop_lcts: set={13,15,16,255}, subset_ratio: float=1.0,
                       margin: int=0, return_ratios: bool=False, rnd_state=None
                      ) -> np.ndarray:
    """
    Makes the filter mask of the values where to extract LCTs from MCD12Q1.
    For more details see the doc of `make_lct_index`. Seperate function
    to use otherwise. 
    """
    # find filepaths
    fpath_0, fpath_1 = umod.corresponding_mcd12q1_files(mcd12q1_fpaths, 
                                                        vh_year=(v,h,year))
    assert fpath_0 is not None, \
        f"Couldn't find MCD12Q1 fpath for year {year}, v {v}, h {h}."
    assert fpath_1 is not None, \
        f"Couldn't find MCD12Q1 fpath for year {year+1}, v {v}, h {h}."
    
    # init datasets 
    mcd12q1_0, mcd12q1_1 = ugeo.ModisDataset(fpath_0), ugeo.ModisDataset(fpath_1)
    
    # get rasters
    lct0 = mcd12q1_0.get_raster("LC_Type1")[0] # There is only one timestep
    lct1 = mcd12q1_1.get_raster("LC_Type1")[0] # in MCD12Q1 files, access it 
    qc0  = mcd12q1_0.get_raster("QC")[0]       # with [0]
    qc1  = mcd12q1_1.get_raster("QC")[0]
    
    # compute masks
    no_change_mask = (lct0 == lct1) & (qc0 == qc1)
    keep_lct_mask  = ~np.isin(lct0, list(drop_lcts)) # a set will be just...
    land_mask      = qc0 == 0                        # ...ignored, must be array-like
    
    if type(rnd_state) is not np.random.RandomState:
        rnd_state = np.random.RandomState(rnd_state)
    random_mask = rnd_state.choice(
        [0, 1], size=lct0.shape, p=[1-subset_ratio, subset_ratio])
    
    if margin is not None and margin > 0:
        margin_mask = np.zeros_like(lct0)
        margin_mask[margin:-margin, margin:-margin] = 1
    else:
        margin_mask = np.ones_like(lct0)
    
    overall_mask = (no_change_mask & keep_lct_mask & 
                    land_mask & random_mask & margin_mask)

    ratios = {
        "no_change_mask": uetc.true_ratio(no_change_mask),
        "keep_lct_mask":  uetc.true_ratio(keep_lct_mask),
        "land_mask":      uetc.true_ratio(land_mask),
        "random_mask":    uetc.true_ratio(random_mask),
        "margin_mask":    uetc.true_ratio(margin_mask),
        "overall_mask":   uetc.true_ratio(overall_mask)
    } if return_ratios else None

    return overall_mask, ratios, lct0

def make_y_index(mcd64a1_fpaths: Iterable[str], mcd12q1_fpaths: Iterable[str], 
                 drop_neg_ratio: float=0.0, rnd_state=None, 
                 check_qa: bool=True, n_workers: int=8, **kwargs):
    """
    Returns:
        y_index, burn_stats, exceptions
    """
    # rnd state
    if type(rnd_state) is not np.random.RandomState:
        rnd_state = np.random.RandomState(rnd_state)

    # set up kwargs that are common among all tasks
    y_index_kwargs = {
        "mcd12q1_fpaths": mcd12q1_fpaths, 
        "drop_neg_ratio": drop_neg_ratio, 
        "check_qa": check_qa, 
        **kwargs
    }
    
    # set up tasks, i.e. tuples: (mcd64a1_fp, seed)
    tasks   = [(fp, uetc.make_seed(rnd_state)) for fp in mcd64a1_fpaths]

    # run jobs
    results = upar.process_tasks(tasks, _make_y_index_worker, 
                                 f_kwargs=y_index_kwargs, n_workers=n_workers)
    
    # split results list
    not_computed = [fp for r, fp in zip(results, mcd64a1_fpaths) if r is None]
    results    = [r for r in results if r is not None]
    exceptions = [r for r in results if isinstance(r[0], Exception)]
    results    = [r for r in results if not isinstance(r[0], Exception)]
    # results: ((y_index_partial, burn_stats_partial), mcd64a1_fpath)
    
    # make dataframes
    y_index = (
        pd.concat([yind for (yind, bs), fp in results])
        .astype({
                "v": np.uint32, "h": np.uint32, "i2": np.uint32, 
                "j2": np.uint32, "year": np.uint32, "yday": np.uint32, 
                "lct": np.uint32, "fire": bool})
    )
    burn_stats = pd.DataFrame({
        os.path.basename(fp): bs for (yind, bs), fp in results
    }).transpose()
    
    return y_index, burn_stats, exceptions, not_computed

def get_undersampling_ratios(burn_stats: pd.DataFrame, 
                             vh_year_weights: Optional[pd.DataFrame]=None
                            ) -> Tuple[float, float]:
    """
    [summary]

    Parameters
    ----------
    burn_stats : dataframe
        [description]
    vh_year_weights : Optional[pd.DataFrame], optional
        Dataframe with identifier columns that are also to be found in 
        `burn_stats`, i.e. one or more of v, h, and year. Must have one column 
        `weight` which holds the weights to be used in the calculation of 
        `r_pos` and `r_pos_us`. `burn_stats` will be (inner-)joined on the 
        columns v, h, and year (not all have to exist in vh_year_weights).

    Returns
    -------
    float, float
        r_pos, r_pos_us
    """
    if vh_year_weights is not None:
        illegal_cols = [col for col in vh_year_weights.columns 
                        if col not in ["v","h","year","weight"]]
        if len(illegal_cols) > 0:
            raise ValueError(f"unallowed columns in vh_year_weights: {illegal_cols}")
        
        common_columns = list(set(burn_stats.columns) & set(vh_year_weights.columns))
        burn_stats = burn_stats.copy() # prevent side effects
        burn_stats = pd.merge(burn_stats, vh_year_weights, 
                              how="inner", on=common_columns)
        burn_stats = burn_stats.assign(
            n_pos = lambda df: df["n_pos"]*df["weight"],
            N     = lambda df: df["N"]    *df["weight"],
            n_pos_us = lambda df: df["n_pos_us"]*df["weight"],
            N_us     = lambda df: df["N_us"]    *df["weight"])
    
    r_pos    = burn_stats["n_pos"].sum() / burn_stats["N"].sum()
    r_pos_us = burn_stats["n_pos_us"].sum() / burn_stats["N_us"].sum()
    return r_pos, r_pos_us
    
def _make_y_index_worker(fp_seed: Tuple[str,int], **kwargs):
    mcd64_fp, seed = fp_seed
    rng = np.random.RandomState(seed)
    try:
        result = make_y_index_for_single_mcd64a1_file(
            mcd64_fp, return_burn_stats=True, rnd_state=rng, **kwargs)
    except Exception as e:
        result = e
    return (result, mcd64_fp)

def make_y_index_for_single_mcd64a1_file(
        mcd64a1_fpath: str, mcd12q1_fpaths: Iterable[str], 
        drop_neg_ratio: float=0.0, rnd_state=None, 
        check_qa: bool=True, return_burn_stats: bool=False, **kwargs
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame,Dict[str,float]]]:
    """
    Also contains respective rows of LCT-index.
    
    Args: 
        drop_negatives_ratio (float):
            For undersampling. Ratio of negative samples to drop
        return_burn_stats (bool, opt):
            If True, returns as second element a dictionary with some stats, 
            e.g. n_pos (originally), r_pos, n_pos_us (undersampled), etc.
        **kwargs:
            Passed to `make_lct_index`. `return_ratios` must not be set to True.
            
    Returns:
        df with cols 'year', 'v', 'h', 'i', 'j', 'lct', 'fire', 'yday'
        i and j are of 500m resolution
        lct is IGBP
    """
    # init MCD64A1 dataset
    mcd64 = ugeo.ModisDataset(mcd64a1_fpath)
    if mcd64.product != "MCD64A1":
        raise ValueError("mcd64a1_fpath must refer to a MCD64A1 file")
        
    v, h = mcd64.vh
    year = mcd64.datetimes[0].year
    
    # get MCD64A1 rasters ([0] since there is only one timestep)
    bd  = mcd64.get_raster("Burn Date")[0]
    bdu = mcd64.get_raster("Burn Date Uncertainty")[0]
    qa  = mcd64.get_raster("QA")[0]
    fd  = mcd64.get_raster("First Day")[0]
    ld  = mcd64.get_raster("Last Day")[0]
    
    # rnd state
    if type(rnd_state) is not np.random.RandomState:
        rnd_state = np.random.RandomState(rnd_state)
    
    # make lct index
    lct_index = make_lct_index(
        v=v, h=h, year=year, mcd12q1_fpaths=mcd12q1_fpaths, 
        rnd_state=rnd_state, **kwargs)
    
    y_index = (
        lct_index.assign(
            burn_date = lambda df: bd[df["i2"], df["j2"]],
            burn_date_uncertainty = lambda df: bdu[df["i2"], df["j2"]],
            burn_first_day = lambda df: fd[df["i2"], df["j2"]],
            burn_last_day  = lambda df: ld[df["i2"], df["j2"]])
        .query("burn_date >= 0")
        .query("burn_date_uncertainty <= 1")
    )
    
    # make fire labels
    y_index.loc[:,"fire"] = y_index["burn_date"] > 0
    
    if return_burn_stats:
        burn_stats = {
            "v": v, "h": h, "year": year, 
            "n_pos": y_index["fire"].sum(),
            "N":     len(y_index)
        }
        burn_stats["r_pos"] = (burn_stats["n_pos"] / burn_stats["N"]
                               if burn_stats["N"] > 0 else np.nan)
        
    # undersample, i.e. keep all positives, drop part of the negatives
    y_index = (
        y_index.assign(
            _fire = lambda df: df["burn_date"] > 0,
            _keep = lambda df: rnd_state.choice(
                [False,True], size=len(df), p=[drop_neg_ratio, 1-drop_neg_ratio]))
        .query("_keep | _fire").drop(columns=["_keep","_fire"])
    )
    
    # check MCD64A1 Quality Assurance values
    if check_qa:
        y_index = (
            y_index.assign(
                _qa    = lambda df: qa[df["i2"], df["j2"]], 
                _qa_ok = lambda df: [_mcd64a1_qa_is_ok(qa) for qa in df["_qa"]])
            .query("_qa_ok").drop(columns=["_qa","_qa_ok"])
        )
    
    # RANDOM DAYS FOR NEGATIVE SAMPLES
    # --------------------------------
    # for all unburned pixels, pick a day randomly within "first" and "last day"
    # (first day and last day mark the timespan for which the MCD64A1 data is
    # reliable)
    month = mcd64.datetimes[0].month
    
    # Sometimes last day is a small number and first day a big one, e.g. 2 and 
    # 360. Then 2 means (I assume) "day 2 of next year".
    first_days_raw = y_index["burn_first_day"].copy()
    last_days_raw  = y_index["burn_last_day"].copy()
    if month > 6: # kinda heuristic. In the end, no random days outside 
                  # the month are generated. first_day > last_day was also only
                  # observed for months 1 and 12.
        # last day is in next year
        sel = first_days_raw > last_days_raw
        ndays_in_current_year = 366 if calendar.isleap(year) else 365
        y_index.loc[sel, "burn_last_day"] += ndays_in_current_year
    else:
        # first day is in previous year
        sel = first_days_raw > last_days_raw
        ndays_in_previous_year = 366 if calendar.isleap(year-1) else 365
        y_index.loc[sel, "burn_first_day"] -= ndays_in_previous_year

    # only draw days which are within the month of the MCD64A1 file
    first_date, last_date = uetc.first_and_last_date_of_month(year, month)

    lower_lims = np.maximum(first_date.day_of_year, y_index["burn_first_day"])
    upper_lims = np.minimum(last_date.day_of_year, y_index["burn_last_day"]) 

    # for the rare case of lower_lims > upper_lims (observed for month 12: 
    # first_day = last_day = 1 => lower_lim = 335 (01.12. in that year))
    keep    = lower_lims <= upper_lims
    y_index = y_index[keep]

    # ... no illegal ydays are left (no yday such as 370 or -1)
    rnd_day_within = rnd_state.randint(lower_lims[keep], upper_lims[keep]+1)
    
    # combine dates of observed fires and randomly picked dates of no-fire
    y_index.loc[:, "yday"] = [
        bd if f else rd for bd,f,rd in 
        zip(y_index["burn_date"], y_index["fire"], rnd_day_within)
    ]
    
    # FINAL STUFF
    # -----------
    # drop columns that are not needed anymore
    y_index.drop(columns=["burn_date","burn_date_uncertainty","burn_first_day",
                          "burn_last_day"], inplace=True)
    
    if return_burn_stats:
        burn_stats["n_pos_us"] = y_index["fire"].sum()
        burn_stats["N_us"]     = len(y_index)
        burn_stats["r_pos_us"] = (burn_stats["n_pos_us"] / burn_stats["N_us"] 
                                  if burn_stats["N_us"] > 0 else np.nan)
        
        return y_index, burn_stats
    else:
        return y_index

def _mcd64a1_qa_is_ok(qa: int) -> bool:
    """
    Args:
        qa (int): value from QA layer in MCD64A1
        
    Returns:
        (bool)
    
    Details:
        https://lpdaac.usgs.gov/documents/875/MCD64_User_Guide_V6.pdf, p. 9
    """
    bits: str = np.binary_repr(qa, width=8)
    
    if bits[-1] == 0: # "bit 0 in doc."
        return False # water cell
    if bits[-2] == 0: # "bit 1"
        return False # valid data flag not true
    if bits[-3] == 1: # "bit 2"
        return False # mapping period shortened
    
    # bits 3 and 4 not of interest
    
    scc = int(bits[:3], 2) # special condition code
    if scc in [1,2,3,4,5]:
        return False
    return True

def get_population_density(lons: Iterable[float], lats: Iterable[float], 
                           year: int, gpw4_dir: str, 
                           raise_incompl: bool=True) -> np.ndarray:
    """
    Reads popultaion densities from GPW4 data. If there is no file for the 
    requested year, the population densities will be interpolated from the year
    before and the one after. 
    
    Args:
        lons, lats (float): 
            longitutes and latitudes (epsg:4326).
        year (int): 
            year of data to extract. One year for all lats and lons.
        gpw4_dir (str): 
            directory containing .tif files that are readable by 
            ugeo.GPW4Dataset and for which the year is extractable by regex 
            `[12][0-9]{3}`, e.g. `gpw_v4_population_density_rev11_2020_30_sec.tif`.
        raise_incompl (bool):
            Raises a FileNotFoundError, if for any of the following year a file
            was not found: 2000, 2005, 2010, 2015, 2020.
            
    Raises:
        FileNotFoundError: see Args
    """
    gds_0, gds_1, year_0, year_1 = _get_popdens_geodatasets(
        year=year, gpw4_dir=gpw4_dir, raise_incompl=raise_incompl
    )
    popdens = _get_popdens_from_geodatasets(
        lons=lons , lats=lats , year=year, 
        gds_0=gds_0, gds_1=gds_1, year_0=year_0, year_1=year_1)

    return popdens

def _get_popdens_geodatasets(
        year: int, gpw4_dir: str, raise_incompl: bool=True, **gds_kwargs
    ) -> Tuple["ugeo.GPW4Dataset", Optional["ugeo.GPW4Dataset"], Optional[int], 
               Optional[int]]:
    """
    """
    fpaths = glob(os.path.join(gpw4_dir, "*.tif"))
    years  = [int(uetc.extract(os.path.basename(fp), "[12][0-9]{3}")) for fp in fpaths]
    years  = np.array(years)
    
    if raise_incompl:
        all_years_of_gpw4 = [2000, 2005, 2010, 2015, 2020]
        for y in all_years_of_gpw4:
            if y not in years:
                raise FileNotFoundError(f"No file found for year {y}")
    
    requires_interpolation = year not in years
    if requires_interpolation:
        # find closest years before and after `year`
        year_before = np.max(years[years < year])
        year_after  = np.min(years[years > year])
        
        fp_before = fpaths[np.where(years == year_before)[0].item()]
        fp_after  = fpaths[np.where(years == year_after )[0].item()]
        
        gds_before = ugeo.GPW4Dataset(fp_before, **gds_kwargs)
        gds_after  = ugeo.GPW4Dataset(fp_after, **gds_kwargs)

        return gds_before, gds_after, year_before, year_after
    else:
        fp  = fpaths[np.where(years == year)[0].item()]
        gds = ugeo.GPW4Dataset(fp, **gds_kwargs)
    
        return gds, None, year, None

def _get_popdens_from_geodatasets(
        lons: Iterable[float], lats: Iterable[float], year: int, 
        gds_0: "ugeo.GPW4Dataset", year_0: Optional[int], 
        gds_1: Optional["ugeo.GPW4Dataset"]=None, year_1: Optional[int]=None
    ) -> np.ndarray:
    """
    """
    lons, lats = np.array(lons), np.array(lats)
    n = len(lats)

    requires_interpolation = gds_1 is not None
    if requires_interpolation:
        var_dtimes_before = [("gpw4", datetime(year_0,1,1))]
        var_dtimes_after  = [("gpw4", datetime(year_1, 1,1))]
        
        popdens_before = ugeo.get_values_from_geodataset(
            gds_0, var_dtimes=var_dtimes_before, xys=(lons,lats))
        popdens_after = ugeo.get_values_from_geodataset(
            gds_1, var_dtimes=var_dtimes_after, xys=(lons,lats))
        
        popdens_before = popdens_before["value"].to_numpy()
        popdens_after  = popdens_after["value"].to_numpy()
        
        # interpolate: weighted avg of year before and year after
        ratio_between = (year-year_0)/(year_1-year_0)
        popdens = popdens_before*(1-ratio_between) + popdens_after*ratio_between
    else:
        var_dtimes = [("gpw4", datetime(year,1,1))]
        popdens    = ugeo.get_values_from_geodataset(
            gds_0, var_dtimes=var_dtimes, xys=(lons,lats))["value"].to_numpy()
    
    return popdens

def get_world_development_indicator(countries: Iterable[str], 
                                    years: Iterable[int], indicator_code: str, 
                                    fpath: str) -> np.ndarray:
    """
    Gets values of a World Development Indicator for given country names and 
    years.
    
    Args:
        fpath (str): 
            filepath to `WDIData.csv`.
        countries (Iterable[str]): 
            series of country names in ISO 3166 ALPHA-3 coding.
        years (array-like): 
            series of years to extract data from, same length as `countries`.
        indicator_code (str): 
            code specifying the indicator to be returned as column `value`, 
            e.g. `NY.GDP.PCAP.PP.KD`
    
    Returns:
        indicator_values (numpy.ndarray, float): extracted values
    
    Details:
        Download `WDIData.csv` at 
        https://datacatalog.worldbank.org/dataset/world-development-indicators
    """
    # df with cols: country_long, country_iso3, year, indicator_value
    indicator_data = _get_wdi_data_by_country_year(
        indicator_code=indicator_code, fpath=fpath)
    
    query_data = pd.DataFrame({"country_iso3": countries,
                               "year": years})
    
    joined = pd.merge(query_data, indicator_data, how="left", 
                      on=["country_iso3", "year"])
    
    return joined["indicator_value"]

def _get_wdi_data_by_country_year(fpath: str, indicator_code: str
                                 ) -> pd.DataFrame:
    """
    Read World Development Indicator data and returns one of the indicators 
    as dataframe.
        
    Args:
        fpath (str): filepath to `WDIData.csv`.
        indicator_code (str): code specifying the indicator to be returned as 
            column `value`, e.g. `NY.GDP.PCAP.PP.KD`
            
    Returns:
        indicator_data (pandas.DataFrame): 
            columns: country_long (str), country_iso3 (str), year (int), 
                     indicator_value (float)
    
    Details:
        Download `WDIData.csv` at 
        https://datacatalog.worldbank.org/dataset/world-development-indicators
    """
    wdi = pd.read_csv(fpath)
    
    id_cols    = ['Country Name','Country Code']
    value_cols = [str(year) for year in range(2000,2020)]
    
    indicator_data = (
        wdi
        [wdi["Indicator Code"] == indicator_code]
        [id_cols+value_cols]
        .melt(id_vars=id_cols, var_name="year", value_name="indicator_value")
        .rename(columns={"Country Name": "country_long", 
                         "Country Code": "country_iso3"})
        .astype({"year": int})
        [["country_long","country_iso3","year","indicator_value"]]
    )
    return indicator_data

def get_from_daily_netcdf(lons: Iterable[float], lats: Iterable[float], 
                          year: int, ydays: Iterable[int], var: str, 
                          fpath: str, ndays: int=1, time_var: str="time",
                          auto_flatten: bool=True) -> np.ndarray:
    """
    Reads values from a netCDF file (readable with xarray).

    Args:
        lons, lats (Iterable[int]):
            Longitudes and latitudes (epsg:4326) at which values are to be extracted.
        ydays (Iterable[int]): 
            days-of-year for which the value at the respective lat/lon coordinate
            is supposed to be read. yday \in [1,366], 366 iff `year` is a leap year.
        ndays (int): 
            number of days up to some yday to extract. E.g. if `ndays=2`, the value
            on day `yday` as well as the value of the day before will be extracted. 
            Must be >0, by default 1.
        auto_flatten (bool):
            If `True` and `ndays=1`, the output will be of shape (len(lats),), 
            otherwise (len(lats), ndays)
        
    Returns:
        values (numpy.ndarray):
            Shape (len(lats), ndays) or (len(lats),), depending on `auto_flatten`.
            The first column refers to the values at the respective days-of-year
            (ydays), the j-th column contains the values for `yday-j`. Values for
            `yday-j < 1` are returned numpy.nan.
    """
    if ndays < 1:
        raise ValueError("ndays must be >= 1")
    ydays = np.array(ydays)
    
    # load
    xds = xr.open_dataset(fpath) # data set (one or more vars)
    xda = xds[var] # data array (only one var)
    
    # extract the requested year (in case it's a multi year file)
    year_sel = xda[time_var].dt.year == year
    xda = xda[year_sel]
    if not upd.dt_series_is_every_day_in_one_year(xda[time_var]):
        raise ValueError(f"date series for year {year} is not daily and/or "
                         "does not cover the entire year.")
    
    # translate lat lons to column and row indices
    tf = ugeo.get_transform(fpath)
    rows, cols = ugeo.get_rowcols_for_coords(xs=lons, ys=lats, tf=tf)
    rows, cols = np.array(rows), np.array(cols)
    
    # filter out those rows and cols that are out of index
    max_date_index, max_row, max_col = np.r_[xda.values.shape] - 1
    out_of_world = (rows > max_row) | (cols > max_col) | (rows < 0) | (cols < 0)
    rows[out_of_world] = 0
    cols[out_of_world] = 0
    
    # extract values
    start_day = ydays-ndays
    start_day[start_day < 0] = 0
    
    values = np.full((len(rows), ndays), np.nan)
    raster_3d = xda.values
    for i in range(ndays):
        # dimensions: date, lat, lon
        date_index  = ydays-i-1 # ydays >= 1, indexing >= 0 -> ...-1
        out_of_year = (date_index < 0) | (date_index > max_date_index)
        date_index[out_of_year] = 0
        values[:,i] = raster_3d[date_index, rows, cols]
        values[out_of_year, i] = np.nan
        
    values[out_of_world, :] = np.nan
        
    xds.close()
    
    if auto_flatten and ndays==1:
        return values.flatten()
    else:
        return values

def get_era5(lons: Iterable[float], lats: Iterable[float], 
             year: int, ydays: Iterable[int], era5_var: str, 
             era5_dir: str, ndays: int) -> np.ndarray:
    """
    Wrapper for `get_from_daily_netcdf` that finds the required era5 files 
    automatically. Files must be yearly and have the respective year in their
    filenames as in `era5_var/*yyyy.nc`.
    
    Example
    -------
    era5_dir = "/home/jonas/data/hackathon/era5/2m_temperature/resolution_100"
    values   = get_era5(
        [13.404954], [52.520008], 2016, ydays=[7], 
        era5_var="t2m", era5_dir=era5_dir, ndays=10)
    """    
    # find file
    era5_fp = glob(os.path.join(era5_dir, f"*{year}.nc"))
    assert len(era5_fp) == 1, \
        f"found !=1 ({len(era5_fp)}) era5 files for year {year}"
    era5_fp = era5_fp[0]
    
    values = get_from_daily_netcdf(
        lons=lons, lats=lats, year=year, ydays=ydays, 
        var=era5_var, fpath=era5_fp, ndays=ndays
    )
    
    return values


class FeatureLoader:
    """
    Loads y_index voxels (year,v,h) and appends X data. 
    
    Calls of append-methods are not evaluated until `apply_to` is called 
    ("lazy"), thus FeatureLoader is designed to be set up once for all 
    append-functions, to then iterate over `apply_to` calls. Some append-methods 
    cache data they need (e.g. global rasters), thus the second call of 
    `apply_to` usually is faster.

    Args:
        y_index_root (str):
            Path to directory containing a directory `by_vh_year` with y_index
            pickle files, named like `{year}_h{h}v{v}.pickle`, e.g. 
            `2002_h01v10.pickle`.

    Attributes:
        callstack:
            Contains all append-functions to be run incl. arguments in the order
            they were called. These calls are evaluated by calling `apply_to`. 
            The callstack is preserved over calls of `apply_to` to be run on a
            different v,h-tile again with the same settings.
        timings: 
            List of most recent timings of evaluations of append-calls, i.e.
            durations in seconds. Empty before the first call to `apply_to`. 
    """
    def __init__(self, year: int, y_index_root: str, verbose: bool=False):
        self.year = year
        self.y_index_root = y_index_root
        self.verbose = verbose
        
        # make properties
        self._v = None
        self._h = None
        self._y_index_fpath = None
        self._y_index = None
        self._Xy = None
        
        # init callstack
        # Each call of an append function will put the passed arguments along
        # with the name of the function into the callstack. Whenever .apply_to
        # is called, all the (real) functions in the callstack will be 
        # executed
        self.callstack: List[str,Dict[str,Any]] = []
        self.timings: List[Tuple[str, float]] = []
        
        # caches
        # (when adding a new cache, add it to empty_caches() as well)
        self._wdi_cache  = {} # key: (fpath, indicator_code)
        self._gpw4_cache = {} # key: gpw4_dir
        self._rpd_cache  = {} # key: (tp_dir, n_climate, n_deficit, month)
        self._ld_cache   = {} # key: (var_name, fpath, hash of kwargs)
        self._grip4_cache = {} # key: (fpath, rename)

    def empty_caches(self) -> "FeatureLoader":
        """
        Empties all caches. Might help when pickling a FeatureLoader instance.
        """
        self._wdi_cache  = {} # key: (fpath, indicator_code)
        self._gpw4_cache = {} # key: gpw4_dir
        self._rpd_cache  = {} # key: (tp_dir, n_climate, n_deficit, month)
        self._ld_cache   = {} # key: (var_name, fpath, hash of kwargs)
        self._grip4_cache = {} # key: (fpath, rename)
        
    @property
    def v(self) -> int:
        if self._v is None:
            raise Exception("call .apply_to(...) first to set v and h")
        return self._v
    
    @property
    def h(self) -> int:
        if self._h is None:
            raise Exception("call .apply_to(...) first to set v and h")
        return self._h
    
    @property
    def y_index_fpath(self) -> str:
        if self._y_index_fpath is None:
            raise Exception("call .apply_to(...) first to set v and h")
        return self._y_index_fpath
    
    @property
    def y_index(self) -> str:
        if self._y_index is None:
            raise Exception("call .apply_to(...) first to set v and h")
        return self._y_index
    
    @property
    def Xy(self) -> str:
        if self._Xy is None:
            raise Exception("call .apply_to(...) first to set v and h")
        return self._Xy
    
    def apply_to(self, v: int, h: int):
        self._v, self._h = v, h
        self._y_index_fpath = os.path.join(
            self.y_index_root, "by_vh_year", 
            f"{self.year}_{self.h__v__}.pickle")
        self._y_index = pd.read_pickle(self.y_index_fpath)
        self._ensure_correct_y_index_format()
        
        # init Xy
        self._Xy = self._y_index

        # convert year and day-of-years to dates (e.g. append_rpd requires that)
        self.dates: pd.DatetimeIndex = uetc.year_day_to_timestamp(
            self.Xy["year"], self.Xy["yday"], always_as_iterable=True)
        
        # convert v,h,i,j to lat,lon
        self.lats, self.lons180 = umod.navigate_inverse(
            v=self.y_index["v"], h=self.y_index["h"], 
            i=self.y_index["i2"], j=self.y_index["j2"], 
            res=2, radians=False)
        self.lons360 = ugeo.lon180_to_lon360(self.lons180)
        
        # process callstack
        self.timings = [] # for very simple profiling
        for funcname, args, kwargs in tqdm(self.callstack, disable=not self.verbose):
            f = getattr(self, funcname)
            t_start = time.time()
            f(*args, **kwargs, __lazy__=False)
            self.timings.append((funcname, time.time()-t_start))
        
        return self

    def _lazy(f):
        """
        Decorator #todo doc
        _check_paths_in_kwargs only kwargs
        """
        # if "fpath" or "gpw4_dir"... in kwargs: check existence

        @wraps(f) # keeps signature, docstr, and name (and maybe more) of 
                  # decorated function
        def g(self, *args, **kwargs):
            __lazy__ = kwargs.pop("__lazy__", True)
            if __lazy__:
                self._check_paths_in_kwargs(kwargs) # not that lazy
                self.callstack.append((f.__name__, args, kwargs)) # very lazy
                return self # allow method chaining
            else:
                return f(self, *args, **kwargs)
        return g

    @staticmethod
    def _check_paths_in_kwargs(kwargs):
        # if there is some kwarg pointing to a dir or file, check if it 
        # exists
        for k in ["fpath","country_shapes_fpath","gpw4_dir","tp_dir",
                  "mcd64a1_dir","mcd12q1_dir","var_dir"]:
            if k in kwargs:
                path = kwargs[k]
                if not os.path.exists(path):
                    raise ValueError(f"{k} doesn't exist {path}")
    
    @_lazy
    def append_era5(self, var_name: str, var_dir: str, lon180: bool, 
                    ndays: int, agg_start: int, rename: Optional[str]=None
                   ) -> "FeatureLoader":
        """
        Args:
            var_name (str): variable name in era5 file.
            ndays (int): number of days from label day back to extract.
            agg_start (int): number of days before label day to start 
                aggregated period.
        """
        # extract values
        X = get_era5(lons=self.lons180 if lon180 else self.lons360, 
                     lats=self.lats, ydays=self.Xy["yday"], 
                     ndays=ndays, year=self.year, era5_var=var_name, 
                     era5_dir=var_dir)
        
        # make features
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # ignore warnings on all-nan rows
            X = feat.agg_tail(X, agg_start=agg_start)
            
        # column names
        var_name = var_name if rename is None else rename
        col_names  = [f"x_{var_name}_{i}" for i in range(agg_start)]
        col_names += [f"x_{var_name}_hist_min", 
                      f"x_{var_name}_hist_mean", 
                      f"x_{var_name}_hist_max"]
        X = pd.DataFrame(X, columns=col_names)
        
        self._append(X)
        return self
    
    @_lazy
    def append_fwi(self, fpath: str, lon180: bool, rename: Optional[str]=None
                  ) -> "FeatureLoader":
        X = get_from_daily_netcdf(lons=self.lons180 if lon180 else self.lons360, 
                                  lats=self.lats, ydays=self.Xy["yday"],
                                  year=self.year, var="fwi", fpath=fpath)
        X = pd.DataFrame({"x_fwi" if rename is None else rename: X})
        
        self._append(X)
        return self
    
    @_lazy
    def append_country_and_continent(self, country_shapes_fpath: str
                                    ) -> "FeatureLoader":
        """
        Args:
            country_shapes_fpath (str):
                Might be "{data_root}/map_features/countries/
                ne_10m_admin_0_countries.shp"
        """
        X = ugeo.country_names_from_lat_lons(
            lats=self.lats, lons=self.lons180, 
            country_shapes_fpath=country_shapes_fpath, 
            cols_to_return=["ISO_A3","CONTINENT"]
        ).rename(columns={"ISO_A3": "country", "CONTINENT": "continent"})
        
        self._append(X)
        return self
    
    @_lazy
    def append_gpw4(self, gpw4_dir: str, rename: Optional[str]=None
                   ) -> "FeatureLoader":
        # check cache and fill it if it is empty 
        if gpw4_dir not in self._gpw4_cache.keys():
            gds_0, gds_1, year_0, year_1 = _get_popdens_geodatasets(
                year=self.year, gpw4_dir=gpw4_dir, raise_incompl=True,
                cache_raster=True
            )
            self._gpw4_cache[gpw4_dir] = {
                "gds_0": gds_0, "gds_1": gds_1, "year_0": year_0, 
                "year_1": year_1
            }
        
        # get values
        X = _get_popdens_from_geodatasets(
            lons=self.lons180 , lats=self.lats , year=self.year, 
            **self._gpw4_cache[gpw4_dir])
        X = (
            pd.Series(X)
            .rename("x_" + ("gpw4" if rename is None else rename))
            .to_frame()
        )
        
        self._append(X)
        return self
    
    @_lazy
    def append_wdi(self, indicator_code: str, fpath: str, 
                   rename: Optional[str]=None) -> "FeatureLoader":
        # prevent unintelligible error message
        if "country" not in self.Xy.columns:
            raise Exception("call append_country_and_continent before "
                            "calling append_wdi or have country codes "
                            "in y_index.")
            
        # check cache and fill it if it is empty
        cache_key = (fpath, indicator_code)
        if cache_key not in self._wdi_cache:
            self._wdi_cache[cache_key] = _get_wdi_data_by_country_year(
                indicator_code=indicator_code, fpath=fpath
            ).query("year == @self.year")
            
        # lookup values
        countries, years = self.Xy["country"], self.Xy["year"]
        query_data = pd.DataFrame({"country_iso3": countries,
                                   "year": years})
        
        X = (
            pd.merge(query_data, self._wdi_cache[cache_key], how="left", 
                     on=["country_iso3", "year"])
            ["indicator_value"]
            .rename(f"x_{rename if rename is not None else indicator_code}")
            .to_frame()
        )
        
        self._append(X)
        return self

    @_lazy
    def append_rpd(self, tp_dir: str, n_climate: int, n_deficit: int, 
                   lon180: bool, rename: Optional[str]=None, **kwargs
                  ) -> "FeatureLoader":
        """
        Appends relative precipitation deficit. 

        Args:
            rename (str, optional):
                RPD-column will be named `x_{rename}` if `rename` is passed, 
                otherwise `x_rpd`.
            **kwargs: passed to `RelativePrecipitationDeficit`
        """
        # RPD is computed for each month and year, thus obtain months (there
        # is only one year: self.year)
        months = np.atleast_1d(self.dates.month) # in case there is only one month

        # create a view on Xy with an additional column to restore order later,  
        # since that order will be lost after groupby
        Xy = self.Xy.assign(_original_order = lambda df: range(len(df)))

        values = [] # will hold one series for each month
        for mo, grp in Xy.groupby(months, sort=False): # sort=False => speed up
            # check cache and fill it if it is empty:
            # for each month in the data, construct one RPD instance 
            # (by def of FeatureLoader there is only one year)
            cache_key = (tp_dir, n_climate, n_deficit, mo)
            if cache_key not in self._rpd_cache.keys():
                self._rpd_cache[cache_key] = RelativePrecipitationDeficit(
                    tp_dir=tp_dir, year=self.year, month=mo, n_climate=n_climate,
                    n_deficit=n_deficit, **kwargs
                )
            rpd: RelativePrecipitationDeficit = self._rpd_cache[cache_key]

            # convert v,h,i,j to lat,lon (can't use self.lons... etc. because 
            # they are not grouped by month)
            lats, lons180 = umod.navigate_inverse(
                v=grp["v"], h=grp["h"], i=grp["i2"], j=grp["j2"], 
                res=2, radians=False)
            lons = lons180 if lon180 else ugeo.lon180_to_lon360(lons180)

            values.append(pd.Series(rpd.get_values(lons=lons, lats=lats),
                          index=grp["_original_order"]))

        # concat results and restore original order of rows
        X = pd.concat(values).sort_index().rename(
            "x_" + ("rpd" if rename is None else rename)
        ).to_frame()

        self._append(X)
        return self

    @_lazy
    def append_neighboring_fires(self, mcd64a1_dir: str, distance: int, 
                                 hindsight: int, rename: Optional[str]=None,
                                 **kwargs) -> "FeatureLoader":
        """
        Args:
            rename (str, optional): 
                If None, the output feature will be named `x_nf`.
            **kwargs: 
                passed to `NeighboringFires`
        """
        # Neighboring fires are computed for each month and year individually, 
        # thus obtain months (there is only one year: self.year)
        months = np.atleast_1d(self.dates.month) # in case there is only one month

        # create a view on Xy with an additional column to restore order later,  
        # since that order will be lost after groupby
        Xy = self.Xy.assign(_original_order = lambda df: range(len(df)))

        nf = NeighboringFires( # handles the caching of monthly rasters
            v=self.v, h=self.h, year=self.year, mcd64a1_dir=mcd64a1_dir, 
            distance=distance, hindsight=hindsight, **kwargs)
        
        values = [] # will hold one series for each month
        for mo, grp in Xy.groupby(months, sort=False):
            values.append(pd.Series(
                nf.get_sums(grp["i2"], grp["j2"], grp["yday"], month=mo),
                index=grp["_original_order"]
            ))
        
        # concat results and restore original order of rows
        X = pd.concat(values).sort_index().rename(
            "x_" + ("nf" if rename is None else rename)
        ).to_frame()

        self._append(X)
        return self

    @_lazy
    def append_vegetation_index(self, mod13q1_dir, use_evi: bool=True, 
                                max_days_diff: int=21, 
                                rename: Optional[str]=None, **kwargs
                               ) -> "FeatureLoader":
        """
        Args:
            rename (str, optional): 
                If None, the output features will be named `x_evi_mean`,
                `x_evi_std`, and `x_evi_tdiff` (if `use_evi`==True otherwise 
                `x_ndvi...`).
            **kwargs: 
                passed to `MostRecentModis`
        """
        var = f"250m 16 days {'EVI' if use_evi else 'NDVI'}"
        mrm = MostRecentModis(v=self.v, h=self.h, product_dir=mod13q1_dir, 
                              product="MOD13Q1", var=var, incl_date=False,
                              max_days_diff=max_days_diff, **kwargs)
        
        vi_name = "evi" if use_evi else "ndvi"
        feature_name = vi_name if rename is None else rename

        yidx = self.y_index # make view to work with
        yidx = yidx.assign(
            _date = uetc.year_day_to_timestamp(yidx["year"], yidx["yday"]),
            _original_order = range(len(yidx))
        )
        
        X = []
        for y_date, grp in yidx.groupby("_date", sort=True):
            n_samples = len(grp)

            # upscale (1 pixel-indices => 4 pixel-indices)
            res4_values = []
            i2, j2 = grp["i2"], grp["j2"]
            IJ4    = umod.upscale_indices(i2, j2) # IJ4: 4-tuple of 2-tuples (i, j)
                                                # upper-left, upper-right, ...
            
            # extract values for each of the 4 upscaled pixel-indices
            for k, (i4, j4) in enumerate(IJ4):
                pxl_vals, vi_date = mrm.get_values(i4, j4, y_date)
                res4_values.append(np.atleast_2d(pxl_vals))
            res4_values = np.vstack(res4_values).T # shape (n_samples, 4)

            # aggregate to resolution "i2,j2"
            if vi_date is None:
                nans = np.full(n_samples, np.nan)
            vi_mean = res4_values.mean(axis=1) if vi_date is not None else nans
            vi_std  = res4_values.std(axis=1)  if vi_date is not None else nans
            tdiff   = (y_date - vi_date).days if vi_date is not None else np.nan
            X.append(pd.DataFrame({
                f"x_{feature_name}_mean": vi_mean, 
                f"x_{feature_name}_std": vi_std,
                f"x_{feature_name}_tdiff": tdiff
            }, index=grp["_original_order"]))

        X = pd.concat(X).sort_index().reset_index(drop=True)

        self._append(X)
        return self

    @_lazy
    def append_lct_nearby(self, mcd12q1_dir: str, query_lcts: Iterable[int], 
                          distance: int, name: str, mcd12q1_var: str="LC_Type1"
                         ) -> "FeatureLoader":
        """
        Args:
            name (str): Output column will be named `x_{name}`.
        """
        # year before since MCD12Q1 is always released the year after
        query_date = pd.Timestamp(year=self.year-1, month=1, day=1)

        mrm = MostRecentModis(v=self.v, h=self.h, product_dir=mcd12q1_dir, 
                              product="MCD12Q1", var=mcd12q1_var, 
                              incl_date=True, max_days_diff=400, 
                              neighbors_kwargs={"dist": distance, 
                                                "outside_val": -1})
        
        neighboring_lcts, lct_date = mrm.get_neighbors(
            self.y_index["i2"], self.y_index["j2"], query_date)
        assert lct_date == query_date, \
            f"lct_dates got messed up {self._error_further_info}"

        X = pd.DataFrame({
            f"x_{name}": np.isin(neighboring_lcts, list(query_lcts)).sum(axis=1)
        })

        self._append(X)
        return self

    @_lazy
    def append_lightnings(self, var_name: str, fpath: str, monthly: bool, 
                          lon180: bool=True, rename: Optional[str]=None, 
                          **kwargs) -> "FeatureLoader":
        """
        Appends lightning values from LIS/OTD 0.5 Degree High Resolution Full or
        Monthly Climatology (HRFC or HRMC) data

        Args:
            var_name (str): 
                Variable in netCDF to extract values from. Some options are [1]
                    HRFC_COM_FR: Combined Flash Rate [720x360, float, fl/km2/yr]  
                    HRFC_OTD_FR: OTD Flash Rate [720x360, float, fl/km2/yr]
                    HRFC_LIS_FR: LIS Flash Rate [720x360, float, fl/km2/yr]
            fpath (str):
                Path to netCDF file.
            monthly (bool): 
                If True, the data for `var_name` in the netCDF file is expected 
                to be 3d, lat-lon-month, and the dates of the samples in 
                `y_index` will be used to extract the lightning values for 
                the corresponding months. If False, data is expected to be 
                2d, i.e. lat-lon, and dates will be ignored. 
            lon180 (bool, optional):
                If True, longitudes are expected to go from -180 to 180. If 
                False, they are expected to go from 0 to 360. 
                HRFC data downloaded from [2] in Nov 2021 has -180 to 180, thus 
                by default True.
            rename (str or None, optional):
                Column in Xy will be named `x_{rename}` if `rename` is passed, 
                otherwise `x_lightn`.
            **kwargs: 
                passed to LightningData instance

        Returns:
            self

        Details:
            [1] https://ghrc.nsstc.nasa.gov/uso/ds_docs/lis_climatology/lohrfc_dataset.html
            [2] https://ghrc.nsstc.nasa.gov/pub/lis/climatology/LIS-OTD/HRFC/data/
        """
        # init LightningData or get existing from cache
        cache_key = (var_name, fpath, uetc.dict_to_hash(kwargs))
        if cache_key not in self._ld_cache.keys():
            self._ld_cache[cache_key] = LightningData(
                var_name=var_name, fpath=fpath, **kwargs)
        ld: LightningData = self._ld_cache[cache_key]
        
        # extract values
        x = ld.get_values(
            lons=self.lons180 if lon180 else self.lons360, lats=self.lats, 
            months=self.dates.month if monthly else None)
            
        # column names
        col_name = "x_" + ("lightn" if rename is None else rename)
        X = pd.DataFrame({col_name: x})
        
        self._append(X)
        return self

    @_lazy
    def append_road_density(self, fpath: str, lon180: bool=True, 
                            rename: Optional[str]=None) -> "FeatureLoader":
        """
        Appends road density data from GRIP4 [1] ascii data. Only tested on 
        `grip4_total_dens_m_km2.asc` (total density, all types combined) 
        downloaded from [2].

        Args:
            rename (str or None, optional):
                Column in Xy will be named `x_{rename}` if `rename` is passed, 
                otherwise `x_road`.

        References:
            [1] Meijer, J.R., Huijbegts, M.A.J., Schotten, C.G.J. and Schipper, 
            A.M. (2018): Global patterns of current and future road infrastructure. 
            Environmental Research Letters, 13-064006. Data is available at 
            www.globio.info
            [2] https://www.globio.info/download-grip-dataset
        """
        # init GRIP4Dataset or get existing from cache
        cache_key = (fpath, rename)
        if cache_key not in self._grip4_cache.keys():
            self._grip4_cache[cache_key] = ugeo.GRIP4Dataset(fpath=fpath)
        g4: ugeo.GRIP4Dataset = self._grip4_cache[cache_key]

        lons=self.lons180 if lon180 else self.lons360
        col_name = "x_" + ("road" if rename is None else rename)
        X = (ugeo.get_values_from_geodataset(g4, xys=(lons, self.lats))
             ["value"].rename(col_name).to_frame())

        self._append(X)
        return self

    def to_pickle(self, Xy_by_vh_year_dir: str) -> None:
        fp = os.path.join(
            Xy_by_vh_year_dir, 
            f"{self.year}_{self.h__v__}.pickle")
        self.Xy.to_pickle(fp, protocol=4)
    
    def _append(self, df: pd.DataFrame) -> None:
        if any([col in self.Xy.columns for col in df.columns]):
            raise ValueError("One or more columns have the same name. Use "
                             "rename arguments to prevent that.")
        self._Xy = pd.concat([self.Xy, df], axis=1)
        
    def _ensure_correct_y_index_format(self) -> None:
        self._y_index = (
            self.y_index
            .reset_index(drop=True)
            .astype({
                "v": np.uint32, "h": np.uint32, "i2": np.uint32, 
                "j2": np.uint32, "year": np.uint32, "yday": np.uint32, 
                "lct": np.uint32, "fire": bool})
        )
        assert len(self.y_index["year"].unique()) == 1, \
            "y_index file contains more than one year, must be only one"

    @property
    def _error_further_info(self) -> str:
        return f"v: {self.v}, h: {self.h}, year: {self.year}"
    
    @property
    def h__v__(self) -> str:
        return umod.h__v__(h=self.h, v=self.v)


class RelativePrecipitationDeficit:
    """
    Args:
        tp_dir (str):
            Directory containing netCDF files of precipitation. In order to 
            not load all files in the directory, 
                1) all filenames must have the format {yyyy}.nc, e.g. "2010.nc" 
                   for the netCDF containing all data for year 2010, and
                2) `filter_fpaths` must be set to True.
        n_climate (int): 
            Number of years over which monthly means will be computed.
        n_deficit (int): 
            Number of years over which deficits will be summed.
        filter_fpaths (bool): 
            If True, only files which are named after those years that are 
            required to compute RPD will be loaded. Thus this saves memory. 
            Set to False if the filenames are not `{yyyy}.nc`.
        var (str, optional): 
            Name of variable to extract in netCDF files. By default "tp".
            
    Properties:
        t_now: 
            {self.year}-{self.month}-01
        t_start_actual: 
            First day of timespan over which the deficits will be summed.
        t_start_climate:
            First day of timespan which will be used to compute the monthly 
            means and thus expected rainfall values.
        tf (Affine):
            from `ugeo.get_transform`
    """
    def __init__(self, tp_dir: str, year: int, month: int, n_climate: int, 
                 n_deficit: int, var: str="tp", filter_fpaths: bool=True):
        self.tp_dir = tp_dir
        self.year  = year
        self.month = month
        self.n_climate = n_climate
        self.n_deficit = n_deficit
        self.var       = var
        self.filter_fpaths = filter_fpaths
        
        self._get_fpaths()
        self._make_timestamps()
        self._get_transform()
        self._compute_deficit_map()
        
    def get_values(self, lons, lats, print_rowcols: bool=False, 
                   return_nans_for_oob_idx: bool=True
                  ) -> Union[np.ndarray, "xr.DataArray"]:
        """
        Args:
            print_rowcols (bool): 
                If True, rows and cols to extract values from ._rpd are printed.
                This is for debugging, e.g. call `rpd._rpd[rows[:1], cols[:1]]` 
                to get an xarray.DataArray which has the lon and lat info 
                corresponding to those rows and cols still attached.
            return_nans_for_oob_idx (bool):
                If True, `np.nan` is returned as value for row- and column-
                indices that are out-of-bounds, i.e. `< 0` or `> max_idx`. 
                If False, such indices will cause an IndexError to be raised. 
                By default True.
        """
        rows, cols = ugeo.get_rowcols_for_coords(xs=lons, ys=lats, tf=self.tf)
        if print_rowcols:
            print(rows, cols)
        
        if return_nans_for_oob_idx:
            max_row, max_col = np.r_[self._rpd.values.shape] - 1
            rows, cols = np.array(rows), np.array(cols)
            oob = (rows > max_row) | (cols > max_col) | (cols < 0) | (rows < 0)
            rows[oob] = 0
            cols[oob] = 0
            vals      = self._rpd.values[rows, cols]
            vals[oob] = np.nan
        else:
            vals = self._rpd.values[rows, cols]
        return vals.flatten()
    
    def _get_fpaths(self):
        self.fpaths = glob(os.path.join(self.tp_dir, "*.nc"))
        self.fpaths = [fp for fp in self.fpaths if self._fname_is_year(fp)]
        
        if self.filter_fpaths:
            self.fpaths = [fp for fp in self.fpaths if 
                           self._year_needed(self._year_from_fpath(fp))]
        
    def _make_timestamps(self):
        self.t_now = pd.Timestamp(year=self.year, month=self.month, day=1)
        self.t_start_actual  = pd.Timestamp(year=self.year-self.n_deficit, 
                                            month=self.month, day=1)
        year_start_climate = self.year-self.n_climate-self.n_deficit
        self.t_start_climate = pd.Timestamp(year=year_start_climate, 
                                            month=self.month, day=1)
        
    def _compute_deficit_map(self):
        xds = xr.concat([xr.open_dataset(fp) for fp in self.fpaths], dim="time")
        climate = xds[self.var][(self.t_start_climate <= xds.time)
                                & (xds.time < self.t_start_actual)]
        climate = climate.groupby("time.month").mean()
        actual  = xds[self.var][(self.t_start_actual <= xds.time)
                                & (xds.time < self.t_now)]
        
        # subtract from each climate month the corresponding actual month
        # (=> positive value means "deficit")
        # http://xarray.pydata.org/en/stable/examples/weather-data.html#Calculate-monthly-anomalies
        deficit = climate - actual.groupby("time.month")
        
        # Relative Rainfall Deficit as xarray.DataArray. Extract values with 
        # self.get_values(...)
        self._rpd = deficit.sum("time") / (climate.sum("month") * self.n_deficit)
    
    @staticmethod
    def _fname_is_year(fp: str):
        return uetc.like(os.path.basename(fp), "^(19|20)[0-9]{2}.nc")
    
    @staticmethod
    def _year_from_fpath(fp: str):
        return int(os.path.basename(fp)[:4]) # fnames in format yyyy.nc
    
    def _year_needed(self, year: int):
        return (year-self.n_climate-self.n_deficit <= year < self.year+1)
    
    def _get_transform(self):
        self.tf   = ugeo.get_transform(self.fpaths[0])
        tfs_equal = [ugeo.get_transform(fp)==self.tf for fp in self.fpaths]
        assert all(tfs_equal), "Not all netCDF files have the same transform"


class NeighboringFires:
    """
    Args:
        hindsight (int, optional):
            A fire in a neighbor-pixel is considered a neighboring fire if it 
            occured within `hindsight` days of the query day.
        distance (int, optional):
            neighborhood distance, passed to `uetc.get_2d_neighbors`, by 
            default 1.
        max_bduc (int): 
            Max days of uncertainty, i.e. max. values in Burn Date Uncertainty 
            of MCD64A1.
    """
    def __init__(self, v: int, h: int, year: int, mcd64a1_dir: str, 
                 hindsight: int=2, distance: int=1, max_bduc: int=2,
                 assert_within_edges: bool=True):
        self.v, self.h, self.year = v, h, year
        self.mcd64a1_dir = mcd64a1_dir
        self.hindsight = hindsight
        self.distance  = distance
        self.max_bduc  = max_bduc
        
        self.mcd64a1_fpaths = glob(os.path.join(mcd64a1_dir, "*", "MCD64A1*.hdf"))
        
        self.raster_cache = {} # key: month
        self.nan = 9999 # Use instead of np.nan to retain int as dtype. 
                        # As ydays only go up to 366, this will not be 
                        # considered a neighboring fire
    
    def get_sums(self, i2: Union[int, Iterable[int]], 
                 j2: Union[int, Iterable[int]], 
                 ydays: Union[int, Iterable[int]], 
                 month: int):
        """
        Args:
            ydays (array-like of int):
                For each pixel (i2,j2) the day of the year for which to check
                for neighboring fires. 
        
        Details:
            Alhough `month` could be inferred from `ydays`, this function only 
            works if all `ydays` are of the same month, thus pass it.
            
            Fires that occur on the same day as the query are not counted.
        """
        ydays  = np.reshape(list(ydays), (len(ydays),1)) # shape (ndays,1)
        i2, j2 = np.atleast_1d(i2), np.atleast_1d(j2)
        
        if month not in self.raster_cache:
            self.raster_cache[month] = self._get_burn_date_raster(month)
        if month-1 not in self.raster_cache: # month=0 => year before, December
            self.raster_cache[month-1] = self._get_burn_date_raster(month-1)
        
        bd0 = self.raster_cache[month-1]
        bd1 = self.raster_cache[month]
        
        wh0 = self._get_bools_for_single_raster(i2, j2, ydays, bd0)
        wh1 = self._get_bools_for_single_raster(i2, j2, ydays, bd1)
        
        # do not count fires twice (although it is unlikely that ij is a fire 
        # in both month0 and month1 within hindsight days)
        wh = wh0 | wh1
        
        # number of fires within hindsight
        return wh.sum(axis=1)
        
    def _get_bools_for_single_raster(self, i2, j2, ydays, bd: np.ndarray
                                    ) -> np.ndarray:
        """
        For each neighboring pixel determine whether there was a fire and if so
        whether it occured within 1-`hindsight` days. 
        
        Returns:
            shape: (len(ydays), number of neighbors)
            type:  bool
        """
        nb_ydays = uetc.get_2d_neighbors(bd, i2, j2, self.distance, 
                                         return_center=False, 
                                         assert_within_edges=True)
        
        days_apart = ydays - nb_ydays # number of days fires occured BEFORE query
                                      # e.g.: 2 => fire occured 2 days before query
        
        within_hindsight = (days_apart > 0) & (days_apart <= self.hindsight)
        return within_hindsight
        
    def _get_burn_date_raster(self, month: int) -> np.ndarray:
        fpath = self._get_fpath(month)
        mcd64 = ugeo.ModisDataset(fpath)
        
        bd = mcd64.get_raster("Burn Date")[0]
        uc = mcd64.get_raster("Burn Date Uncertainty")[0]
        # skip check of QA layer
        
        bd[bd < 1] = self.nan
        bd[uc > self.max_bduc] = self.nan
        
        if month == 0:
            ndays = 366 if calendar.isleap(self.year-1) else 365
            bd   -= ndays # self.nan still large enough
        return bd
    
    def _get_fpath(self, month: int) -> str:
        year  = self.year if month > 0 else self.year-1
        month = month if month > 0 else 12
        pattern = (f"{self.year}.{month:02d}.01/MCD64A1\\.A{self.year}[0-9]+\\."
                   f"h{self.h:02d}v{self.v:02d}\\.006\\.[0-9]+\\.hdf$")
        for fp in self.mcd64a1_fpaths:
            if uetc.like(fp, pattern):
                return fp
        raise RuntimeError(f"Didn't find MCD64A1 file for v: {self.v}, "
                           f"h: {self.h}, year: {self.year}, month: {month}")


class MostRecentModis:
    """
    Only works with MODIS HDF files which only contain rasters for one day.
    
    Args: 
        product (str):
            short name of product, e.g. MOD13Q1 (not MOD13Q1.006)
        var (str):
            Name of the Science Data Set in the MODIS HDF file to extract from.
        incl_date (bool):
            If True, `date` in `get_values` and `get_neighbors` is inclusive, 
            otherwise the moste recent date can be the day before `date`.
        return_none_for_broken_hdfs (bool, optional):
            If True, and the most recent HDF file turns out to be broken, the
            MostRecentModis instance behaves just as if no HDF file was found.

    Details:
        When extracting values for many different dates, do so ordered by date
        since the most recently used raster is cached.
    """
    def __init__(self, v: int, h: int, product_dir: str, product: str, var: str, 
                 incl_date: bool, max_days_diff: Optional[int]=None, 
                 neighbors_kwargs={}, mds_kwargs={}, 
                 return_none_for_broken_hdfs: bool=True):
        self.v, self.h   = v, h
        self.product_dir = product_dir
        self.product     = product
        self.var         = var
        self.incl_date   = incl_date
        self.max_days_diff = max_days_diff
        self.neighbors_kwargs = neighbors_kwargs
        self.mds_kwargs = mds_kwargs
        self.return_none_for_broken_hdfs = return_none_for_broken_hdfs
        
        self._raster_cache = None
        self._mds_cache    = None
        
        self.fpaths = glob(os.path.join(product_dir, "*", f"*{self.h__v__}*.hdf"))
        if len(self.fpaths) < 1:
            raise RuntimeError(f"no  MODIS HDF files found in {product_dir}")
        self.hdf_index = (
            umod.make_hdf_index_from_paths(self.fpaths, path_col_name="fpath")
            .set_index("fname_date")
            .sort_index(ascending=True)
            .query("product == @self.product")
        )
        
        assert self.hdf_index["product"].nunique() == 1, \
            "product_dir contains HDF files of several products; must be one"
    
    def get_values(self, i, j, date: pd.Timestamp
                  ) -> Tuple[np.ndarray, Union[None, pd.Timestamp]]:
        """
        Returns:
            values, most_recent_date
            (if any raster was found, otherwise None, None)
        """
        i, j = np.atleast_1d(i), np.atleast_1d(j)
        raster, most_recent_date = self._get_raster_of_most_recent(date)
        values = raster[i,j] if raster is not None else None
        return values, most_recent_date
    
    def get_neighbors(self, i, j, date: pd.Timestamp
                     ) -> Tuple[np.ndarray, Union[None, pd.Timestamp]]:
        """
        Returns:
            neighbor_values, most_recent_date
            (if any raster was found, otherwise None, None)
        """
        i, j = np.atleast_1d(i), np.atleast_1d(j)
        raster, most_recent_date = self._get_raster_of_most_recent(date)
        
        if raster is None:
            neighbor_values = None
        else:
            neighbor_values = uetc.get_2d_neighbors(
                raster, i, j, **self.neighbors_kwargs)
        return neighbor_values, most_recent_date
            
    def _get_raster_of_most_recent(self, date: pd.Timestamp):
        """
        Returns:
            raster, most_recent_date
            (if any raster was found, otherwise None, None)
        """
        most_recent_meta = self._get_most_recent_hdf_index_row(date)
        if most_recent_meta is None:
            return None, None
        fpath = most_recent_meta.fpath
        
        # check cache
        if self._mds_cache is None or self._mds_cache.fpath != fpath:
            try: # hdf file might be broken
                self._mds_cache = ugeo.ModisDataset(fpath, **self.mds_kwargs)
            except RasterioIOError:
                if self.return_none_for_broken_hdfs:
                    return None, None
                raise
            self._raster_cache = self._mds_cache.get_raster(self.var)[0]
        # else: cache already contains the raster that is looked for
        
        most_recent_date = pd.Timestamp(self._mds_cache.datetimes[0])
        return self._raster_cache, most_recent_date
    
    def _get_most_recent_hdf_index_row(self, date: pd.Timestamp):
        hdfidx = self.hdf_index # make view to work with
        date   = pd.Timestamp(date, freq="D") # ensure days as freq.
        
        if self.max_days_diff is not None:
            min_date = date - (self.max_days_diff * date.freq)
            hdfidx = hdfidx.query("fname_date >= @min_date")
        if self.incl_date:
            date = date + (1*date.freq)
        
        try:
            return hdfidx.query("fname_date < @date").iloc[-1]
        except IndexError:
            # query-result empty, no file fulfills the query conditions
            return None
        
    @property
    def h__v__(self) -> str:
        return umod.h__v__(h=self.h, v=self.v)


class LightningData:
    """
    Reads values from a netCDF file (readable with xarray) from one of the data 
    products 
     * LIS/OTD HRFC, e.g. `LISOTD_HRFC_V2.3.2015.nc` or
     * LIS/OTD HRMC, e.g. `LISOTD_HRMC_V2.3.2015.nc`.

    Args:
        var_name (str): e.g. "HRFC_COM_FR" or "HRMC_COM_FR"
        interp_na (bool, optional):
            #todo
        max_na_gaps (Dict[str, int] or None, optional):
            If None, default values are taken, see details. 
            Ignored if interp_na=False.

    Details:
        LIS/OTD HR{F/M}C data is described here
        https://ghrc.nsstc.nasa.gov/uso/ds_docs/lis_climatology/LISOTD_climatology_dataset.html

        Default max_na_gaps:
            "Month": 2
                fills at most the gaps in e.g. Dec, _, _, Mar
            "Longitude": 60
                30 deg*; seems much, but NAs in HRxC are predominantly in far 
                north or south where 1 deg lon is not much
            "Latitude": 20
                10 deg*; less than for lon because seasonal effects are much 
                stronger on latitude 

            * HRFC and HRMC are published in .5 deg resolution
    """
    def __init__(self, var_name: str, fpath: str, interp_na: bool=True, 
                 max_na_gaps: Optional[Dict[str,int]]=None):
        self.var_name  = var_name
        self.fpath     = fpath
        self.interp_na = interp_na

        if max_na_gaps is None:
            self.max_na_gaps = {
                "Month": 2, # fills at most the gaps in e.g. Dec, _, _, Mar 
                "Longitude": 60, # => 30 deg; seems much, but NAs are 
                                 # predominantly in far north or south where 
                                 # 1 deg lon is not much
                "Latitude": 20 # => 10 deg; here less than for lon also because
                               # seasonal effects are much stronger on latitude 
            }
        else:
            self.max_na_gaps = max_na_gaps

        # load
        self.xds = xr.open_dataset(fpath, decode_times=False) # data set (one or 
                                                              # more vars)
        self.xda = self.xds[var_name] # data array (only one var)
        self.monthly = "Month" in self.xda.dims
        
        if self.interp_na:
            self._interpolated_xdas = {}
            for dim in self.xda.dims:
                xda = self._repeat_along_month_dim(self.xda) \
                      if dim == "Month" else self.xda
                self._interpolated_xdas[dim] = xda.interpolate_na(
                    dim, max_gap=self.max_na_gaps[dim], limit=self.max_na_gaps[dim])

    @staticmethod
    def _repeat_along_month_dim(xda: "xr.DataArray") -> "xr.DataArray":
        # Requires loading xarray (see imports)
        # this will allow interpolating from Dec to Feb (e.g.). Otherwise
        # Jan would be at the end. 
        xda_before = xda.assign_coords(Month = xda["Month"].values - 12)
        xda_after  = xda.assign_coords(Month = xda["Month"].values + 12)
        return xr.concat([xda_before,xda,xda_after], dim="Month")

    def get_values(self, lons: Union[float, Iterable[float]], 
                   lats: Union[float, Iterable[float]], 
                   months: Optional[Union[int, Iterable[int]]]=None
                  ) -> np.ndarray:
        """
        Args:
            lons, lats (array-like of float, or float):
                Longitudes and latitudes (epsg:4326) at which values are to be 
                extracted.
            months (array-like of int, int, or None; optional):
                Whether the data contains a Month-dimension will be inferred 
                at init-time. If so, `get_values` expects an array-like to 
                be passed as `month`. Month values must be 1-based, i.e. 
                Jan = 1, Dec = 12. 
            
        Returns:
            values (numpy.ndarray):
                Shape (len(lats),)

        Dev notes:
            Extracting values by lat lon coords with xarray is shown here
            http://xarray.pydata.org/en/stable/user-guide/indexing.html
            (look for "Latitude")
        """
        # ensure that lons and lats are iterables
        lons, lats = np.atleast_1d(lons), np.atleast_1d(lats)

        # prepare selectors
        sel_kwargs = {"Longitude": xr.DataArray(lons, dims="points"),
                      "Latitude":  xr.DataArray(lats, dims="points")}
        if self.monthly:
            if months is None:
                raise RuntimeError("monthly data has been past, so months must "
                                   "not be None")
            months = np.atleast_1d(months)
            sel_kwargs["Month"] = xr.DataArray(months, dims="points")
        else:
            if months is not None:
                raise RuntimeError("months passed but no monthly data loaded")
        
        # extract
        if self.interp_na:
            values = [xda.sel(**sel_kwargs, method="nearest").values
                      for xda in self._interpolated_xdas.values()]
            values = uetc.hstack_flat_arrays(*values)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") # ignore warnings on all-nan rows
                values = np.nanmean(values, axis=1) # if all nan -> val = nan
        else:
            values = self.xda.sel(**sel_kwargs, method="nearest").values
        return values

    def __del__(self):
        """
        Closes the xarray connection to the netcdf whenever this object is 
        desctructed.
        """
        self.xds.close()

