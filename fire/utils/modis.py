"""
Functions for processing MODIS hdf files and filenames
"""
import os
import datetime
import warnings

from typing import List, Tuple, Optional, Callable, Iterable, Union

import numpy as np
import pandas as pd

from fire.utils.etc import max_precision, extract, like

# CONSTANTS
# ----------------------------------------------------
# from MODIS_C6_FIRE_USER_GUIDE_A.pdf, p. 27
R    = max_precision(6371007.181) # m, the radius of the idealized sphere 
                                  # representing the Earth
T    = max_precision(1111950.5196666666) # m, the height and width of each MODIS 
                                         # tile in the projection plane
XMIN = max_precision(-20015109.354) # m, the western limit of the projection plane
YMAX = max_precision( 10007554.677) # m, the northern limit of the projection plane
W    = max_precision(T/1200)
#W    = T/1200 # = 926.62543305 m, the actual size of a "1-km" 
              # MODIS sinusoidal grid cell.
# ----------------------------------------------------

# CONSTANTS
# ----------------------------------------------------
# from MODIS_C6_FIRE_USER_GUIDE_A.pdf, p. 27
# R    = max_precision(6371007.181) # m, the radius of the idealized sphere 
#                                   # representing the Earth
# T    = max_precision(1111950) # m, the height and width of each MODIS 
#                               # tile in the projection plane
# XMIN = max_precision(-20015109) # m, the western limit of the projection plane
# YMAX = max_precision( 10007555) # m, the northern limit of the projection plane
# W    = max_precision(926.62543305)
#W    = T/1200 # = 926.62543305 m, the actual size of a "1-km" 
              # MODIS sinusoidal grid cell.
# ----------------------------------------------------

# vh tiles for which MODIS files were found (obtained from MODIS URL index)
VH_WHITELIST = [
    (0, 14), (0, 15), (0, 16), (0, 17), (0, 18), (0, 19), (0, 20),
    (0, 21), (1, 12), (1, 13), (1, 14), (1, 15), (1, 16), (1, 17),
    (1, 18), (1, 19), (1, 20), (1, 21), (1, 22), (1, 23), (2, 9),
    (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (2, 15), (2, 16),
    (2, 17), (2, 18), (2, 19), (2, 20), (2, 21), (2, 22), (2, 23),
    (2, 24), (2, 25), (2, 26), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10),
    (3, 11), (3, 12), (3, 13), (3, 14), (3, 15), (3, 17), (3, 18),
    (3, 19), (3, 20), (3, 21), (3, 22), (3, 23), (3, 24), (3, 25),
    (3, 26), (3, 27), (3, 28), (3, 29), (4, 8), (4, 9), (4, 10),
    (4, 11), (4, 12), (4, 13), (4, 14), (4, 17), (4, 18), (4, 19),
    (4, 20), (4, 21), (4, 22), (4, 23), (4, 24), (4, 25), (4, 26),
    (4, 27), (4, 28), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11),
    (5, 12), (5, 15), (5, 16), (5, 17), (5, 18), (5, 19), (5, 20),
    (5, 21), (5, 22), (5, 23), (5, 24), (5, 25), (5, 26), (5, 27),
    (5, 28), (5, 29), (5, 30), (6, 2), (6, 3), (6, 7), (6, 8), (6, 9),
    (6, 10), (6, 11), (6, 16), (6, 17), (6, 18), (6, 19), (6, 20),
    (6, 21), (6, 22), (6, 23), (6, 24), (6, 25), (6, 26), (6, 27),
    (6, 28), (6, 29), (6, 30), (6, 31), (7, 1), (7, 3), (7, 7), (7, 8),
    (7, 9), (7, 10), (7, 11), (7, 12), (7, 15), (7, 16), (7, 17),
    (7, 18), (7, 19), (7, 20), (7, 21), (7, 22), (7, 23), (7, 24),
    (7, 25), (7, 26), (7, 27), (7, 28), (7, 29), (7, 30), (7, 31),
    (7, 32), (7, 33), (7, 34), (8, 0), (8, 1), (8, 2), (8, 8), (8, 9),
    (8, 10), (8, 11), (8, 12), (8, 13), (8, 16), (8, 17), (8, 18),
    (8, 19), (8, 20), (8, 21), (8, 22), (8, 23), (8, 25), (8, 26),
    (8, 27), (8, 28), (8, 29), (8, 30), (8, 31), (8, 32), (8, 33),
    (8, 34), (8, 35), (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 8),
    (9, 9), (9, 10), (9, 11), (9, 12), (9, 13), (9, 14), (9, 16),
    (9, 18), (9, 19), (9, 20), (9, 21), (9, 22), (9, 23), (9, 25),
    (9, 27), (9, 28), (9, 29), (9, 30), (9, 31), (9, 32), (9, 33),
    (9, 34), (9, 35), (10, 0), (10, 1), (10, 2), (10, 3), (10, 4),
    (10, 5), (10, 10), (10, 11), (10, 12), (10, 13), (10, 14),
    (10, 17), (10, 19), (10, 20), (10, 21), (10, 22), (10, 23),
    (10, 27), (10, 28), (10, 29), (10, 30), (10, 31), (10, 32),
    (10, 33), (10, 34), (10, 35), (11, 1), (11, 2), (11, 3), (11, 4),
    (11, 5), (11, 6), (11, 8), (11, 10), (11, 11), (11, 12), (11, 13),
    (11, 14), (11, 15), (11, 19), (11, 20), (11, 21), (11, 22),
    (11, 23), (11, 27), (11, 28), (11, 29), (11, 30), (11, 31),
    (11, 32), (11, 33), (12, 11), (12, 12), (12, 13), (12, 16),
    (12, 17), (12, 19), (12, 20), (12, 24), (12, 27), (12, 28),
    (12, 29), (12, 30), (12, 31), (12, 32), (13, 5), (13, 12),
    (13, 13), (13, 17), (13, 20), (13, 21), (13, 22), (13, 28),
    (13, 29), (13, 30), (13, 31), (14, 13), (14, 14), (14, 15),
    (14, 16), (14, 18), (14, 22), (14, 27), (14, 28), (15, 15),
    (15, 17), (15, 18), (15, 19), (15, 20), (15, 21), (15, 22),
    (15, 23), (15, 24), (16, 14), (16, 15), (16, 16), (16, 17),
    (16, 18), (16, 19), (16, 20), (16, 21), (16, 22), (16, 23),
    (17, 14), (17, 15), (17, 16), (17, 17), (17, 18), (17, 19),
    (17, 20), (17, 21)
]


#todo doesn't work for lat=34.1042, lon=-118.964 (result completely off)
# plausible result vh= 5, 8; row=707, col=179 (obtained with get_fires)
def navigate_forward(lat: float, lon: float, res: int=1, radians: bool=False
                    ) -> Tuple[int,int,int,int]:
    """
    Args:
        lat: latitude of location in radians or degrees, depending on `radians`.
            Variable phi in MODIS-Doc. 
        lon: longitude of location in radians or degrees, depending on `radians`.
            Variable lambda in MODIS-Doc. 
        res: resolution of MODIS product; 
             1 for "1-km", 2 for "500-m", 4 for "250-m"
        radians (bool): If True, input args lat and lon are expected to be 
            in radians. If False, in degrees. By default False.
    
    Returns:
        (v, h, row, col), where 
            v:   vertical coordinate of MODIS tile; 0 ≤ v ≤ 17
            h:   horizontal coordinate of MODIS tile; 0 ≤ h ≤ 35
            row: row in MODIS tile (i in MODIS-Doc), 
                 0 ≤ row ≤ 1199 (at least for MOD14A2)
            col: column in MODIS tile (j in MODIS-Doc), 
                 0 ≤ col ≤ 1199 (at least for MOD14A2)
                     
    Details:
        See section "Forward Mapping" in MODIS_C6_FIRE_USER_GUIDE_A.pdf, p. 27.
        To double-check, see "Tile Calculator Tool" 
        https://landweb.modaps.eosdis.nasa.gov/cgi-bin/developer/tilemap.cgi
        
        * On a random sample very close to output of Tile Calculator Tool
          but might be too far off still for alignment with non-MODIS products
        * check: consistent with MODIS meta data?
        
    Consider: 
        * https://newsroom.gsfc.nasa.gov/sdptoolkit/HEG/HEGHome.html
        * compare to GeoTIFFs!
        * not floor-ing
        * can row (or col) be ==1200 ? Wouldn't this be a neighboring tile?
        
    """
    # get inputs as radians
    if not radians:
        lat, lon = np.deg2rad(lat), np.deg2rad(lon)

    # Adjust w for tile size
    # w should be 231.65635826 for res=4, 
    #             463.31271653 for res=2, 
    #         and 926.62543305 for res=1
    w = W/res
    
    x = R * lon * np.cos(lat)
    y = R * lat
    
    h = int(np.floor( (x-XMIN)/T ))
    v = int(np.floor( (YMAX-y)/T ))
    
    i_nominator = (YMAX-y) % T
    i = int(np.floor( i_nominator/w - 0.5 ))
    
    j_nominator = (x-XMIN) % T
    j = int(np.floor( j_nominator/w - 0.5 ))
    
    return v, h, i, j

def lat_lon_to_vh(lat: float, lon: float) -> Tuple[int,int]:
    """
    Args:
        lat, lon (float): Latitude and longitude, both in degrees.

    Returns:
        tuple (int,int): Tuple of vertical and horizontal tile index of MODIS' 
            sinusoidal grid.
    """
    v, h, _, _ = navigate_forward(lat, lon)
    return v, h

def navigate_inverse(v: int, h: int, i: int, j: int, res: int,
                     radians: bool=False, wrap: bool=True
                    ) -> Tuple[float, float]:
    """
    Args:
        v: vertical coordinate of MODIS tile; 0 ≤ V ≤ 17
        h: horizontal coordinate of MODIS tile; 0 ≤ H ≤ 35
        i: row in MODIS tile (i in MODIS-Doc), 0 ≤ row ≤ 1199
        j: column in MODIS tile (j in MODIS-Doc), 0 ≤ col ≤ 1199
        res: resolution of MODIS product; 1 for "1-km", 2 for "500-m", 
            4 for "250-m"
        radians (bool (opt)): If True, lat, lon will be returned in radians,
            otherwise in degrees. By default False.
        wrap (bool (opt)): If True, longitudes outside of [-180,180] will
            be shifted by 360 degrees, such that they are within [-180,180].
            Default True.
    
    Returns:
        lat, lon: Latitude and longitude of grid cell(s) (at center) in degrees
            (if `radians=False`) or radians (if `radians=True`).
        
    Details:
        See section "Inverse Mapping" in MODIS_C6_FIRE_USER_GUIDE_A.pdf, p. 28.
        To double-check, see "Tile Calculator Tool" 
        https://landweb.modaps.eosdis.nasa.gov/cgi-bin/developer/tilemap.cgi
        
    """
    #todo implement Applicability to 250-m and 500-m MODIS Products, p. 28
    # of MOD14A1 doc / guide

    # passing scalars and arrays at the same time to this function
    # is experimental. Check if that is the case and yield a warning
    inputs_are_arrays = [not np.isscalar(x) for x in [v,h,i,j]]
    if np.any(inputs_are_arrays) and not np.all(inputs_are_arrays):
        warnings.warn("Passing scalars and arrays at the same time "
                      "is experimental. Double check the shape of "
                      "the output or pass values as either (all) arrays "
                      "of the same length or as (all) scalars.")
    
    # to allow for a mixture of scalars and arrays as arguments
    # transform i to the shape of j, if i is a scalar
    # and j is not. Due to the following computations and
    # due to broadcasting in numpy, j does not have to be transformed.
    if np.isscalar(i) and not np.isscalar(j):
        i = np.ones(j.shape) * i
    
    v,h,i,j = np.atleast_1d(v), np.atleast_1d(h), np.atleast_1d(i), np.atleast_1d(j)

    # Adjust w for tile size
    # w should be 231.65635826 for res=4, 
    #             463.31271653 for res=2, 
    #         and 926.62543305 for res=1
    w = W/res
    
    x = (j + 0.5)*w + h*T + XMIN
    y = YMAX - (i + 0.5)*w - v*T
    
    lat = y / R
    lon = x / (R*np.cos(lat))

    #todo lon shouldn't be < -180 (?)
    # got e.g. lat: 9.995874, lon: -182.770235
    # should be lat: (same), lon: 177.23 (?)
    if wrap:
        lon[lon < -180] = lon[lon < -180] + 360
        lon[lon >  180] = lon[lon >  180] - 360
    
    if radians:
        return lat, lon
    else:
        return np.rad2deg(lat), np.rad2deg(lon)

def upscale_indices(i: Union[int, Iterable[int]], 
                    j: Union[int, Iterable[int]]
                   ) -> Tuple[Tuple[int,int], Tuple[int,int], 
                              Tuple[int,int], Tuple[int,int]]:
    """
    Converts ij-indices from some resolution to the next higher one, i.e.
    from `i1,j1` (1000m or 1 pixels/km) to `i2,j2` (500m or 2 pixels/km) or 
    from `i2,j2` (500m or 2 pixels/km) to `i4,j4` (250m or 4 pixels/km).
    
    Returns:
        4-tuple of 2-tuple of numpy arrays, each of length len(i):
            upper_left, upper_right, lower_left, lower_right
    """
    if (len(np.atleast_1d(i)) > 1) or (len(np.atleast_1d(j)) > 1):
        i, j = np.atleast_1d(i), np.atleast_1d(j)
        assert len(i) == len(j), \
            f"i and j must be of equal length, got {len(i)} and {len(j)}"
    i, j = i*2, j*2
    return (i, j), (i, j+1), (i+1, j), (i+1, j+1)

def meta_from_hdf_filename(hdf_fname:str) -> dict:
    """
    Returns meta data that can be inferred from .hdf filenames.
    
    Args:
        hdf_name: 
            filename or -path to an MODIS HDF file. The filename
            must be the original name given by NASA, e.g.
            "MOD14A1.A2019257.h11v12.006.2019269172641.hdf"
            
    Returns:
        dict with key : type(value) : description
            "product" : str : e.g. "MOD14A1" (not "MOD14A1.006")
            "date"    : datetime.datetime : date parsed from filename
            "h"       : int : horizontal tile number
            "v"       : int : vertical tile number
    """
    meta = {
        "product" : product_name_from_hdf_filename(hdf_fname),
        # "sat_name": sat_name_from_hdf_filename(hdf_fname),
        "date"    : date_from_hdf_filename(hdf_fname),
        "h"       : h_from_hdf_filename(hdf_fname),
        "v"       : v_from_hdf_filename(hdf_fname)
    }
    
    return meta

def product_name_from_hdf_filename(hdf_fname:str) -> str:
    hdf_fname = os.path.basename(hdf_fname)
    return extract(hdf_fname, r"^[^.]+")

#todo del
def sat_name_from_hdf_filename(hdf_fname:str) -> str:
    warnings.warn("sat_name_from_hdf_filename deprecated")
    hdf_fname = os.path.basename(hdf_fname) # drop path if it exists
    return hdf_fname[:3]

def date_from_hdf_filename(hdf_fname:str) -> datetime.datetime:
    hdf_fname = os.path.basename(hdf_fname) # drop path if it exists
    # date is given in yyyyjjj, where jjj is day-of-year
    date_str = extract(hdf_fname, r"[12][0-9]{6}")
    return datetime.datetime.strptime(date_str, "%Y%j")

def h_from_hdf_filename(hdf_fname:str) -> int:
    hdf_fname = os.path.basename(hdf_fname) # drop path if it exists
    h = extract(hdf_fname, r"(?<=\.h)[0-9]{2}(?=v[0-9]{2}\.)")
    return int(h)

def v_from_hdf_filename(hdf_fname:str) -> int:
    hdf_fname = os.path.basename(hdf_fname) # drop path if it exists
    v = extract(hdf_fname, r"(?<=h[0-9]{2}v)[0-9]{2}(?=\.)")
    return int(v)

def default_target_path_scheme(url:str, data_root_path:str) -> str:
    """
    Generates a file path to which a file from url will be
    written to. The scheme is as follows:
        {data_root_path}/{product}/{date}/{filename}
        
    E.g. the prameters
        url = 'https://e4ftl01.cr.usgs.gov/MOTA/MCD12Q1.006/2001.01.01/'
              'MCD12Q1.A2001001.h00v08.006.2018142182903.hdf'
        data_root_path = '/home/user/data/'
    will yield
        '/home/user/data/MCD12Q1.006/2001.01.01/'
        'MCD12Q1.A2001001.h00v08.006.2018142182903.hdf'
    """
    data_root_path      = os.path.expanduser(data_root_path)
    product_name        = product_name_from_hdf_filename(url)
    url_from_product_on = extract(url, product_name + r".+[/\\].+")
    target_path         = os.path.join(data_root_path, url_from_product_on)
    return target_path

def make_hdf_index_from_paths(hdf_paths: List[str], 
                              path_col_name: str = "url",
                              drop_duplicates: bool=True
                             ) -> pd.DataFrame:
    """
    Creates a pandas DataFrame with some meta data extracted from the filenames 
    of MODIS HDF files. Thus, the filenames must be given in the original format 
    as provided on LPDAAC, e.g.
        `MCD12Q1.A2001001.h00v08.006.2018142182903.hdf`.

    Args:
        hdf_paths (List[str]): List of paths to MODIS HDF files.
        path_col_name (str (opt)): Name of the column that will hold all the paths 
            passed via `hdf_paths`. Defaults to `url`.
        drop_duplicates (bool): If True, duplicates in `hdf_paths` will be dropped.

    Returns:
        pd.DataFrame: columns:
            {path_col_name} (str): As in arg hdf_paths. The name of this column 
                is passed as arg `path_col_name`.
            fname (str): Filename (or basename) extracted from path in hdf_paths.
            product (str): E.g. `MCD12Q1`
            fname_date (datetime): Datetime object from parsing the date given in 
                the filename after the product name.
            v (int): Vertical tile number from sinusoidal MODIS grid.
            h (int): Horizontal tile number from sinusoidal MODIS grid.
            vh (Tuple[int,int]): Tuple of v and h for convenience when filtering.

        not included anymore:
            sat_name (str): E.g. `MCD` #

    """
    if drop_duplicates:
        hdf_paths = np.unique(hdf_paths)

    hdf_index = (
        pd.DataFrame({path_col_name: hdf_paths})
        .assign(fname   = lambda df: df[path_col_name].apply(os.path.basename),
                product = lambda df: df.fname.apply(product_name_from_hdf_filename),
                fname_date = lambda df: df.fname.apply(date_from_hdf_filename),
                v          = lambda df: df.fname.apply(v_from_hdf_filename),
                h          = lambda df: df.fname.apply(h_from_hdf_filename),
                vh = lambda df: pd.Series(zip(df.v, df.h)) # much faster than apply
        )
    )
    
    return hdf_index

def filename_tastes_like_modis_hdf(fpath: str) -> bool:
    re = r"^[A-Z_0-9]+\.A[12][0-9]{6}\.h[0-9]{2}v[0-9]{2}\.[0-9]{3}\.[0-9]{8,}\.hdf$"
    return like(os.path.basename(fpath), re)

#todo
def hdf_file_is_ok(fpath: str) -> bool:
    """
    Checks if a MODIS HDF file is ok, by asserting that 
    1) the file is readable,
    2) dates are consistent with ____ #todo Dominiks code / hints
    3) there are 8 dates (if mod14a1) (optional)
    4) v and h from file name is consistent with inner file meta
    5) ...?
    """
    pass

def get_corresponding_mcd12q1_files(modis_file: str, mcd12q1_files: List[str]
                                   ) -> Tuple[str,str]:
    warnings.warn("Deprecated function, use `corresponding_mcd12q1_files` instead",
                  UserWarning)
    warnings.warn("Deprecated function, use `corresponding_mcd12q1_files` instead",
                  DeprecationWarning)
    return corresponding_mcd12q1_files(modis_file=modis_file, 
                                       mcd12q1_files=mcd12q1_files)

def corresponding_mcd12q1_files(mcd12q1_files: List[str], 
                                modis_file: Optional[str]=None, 
                                vh_year: Optional[Tuple[int,int,int]]=None
                               ) -> Tuple[str,str]:
    """
    Returns:
        tuple of (str, str): First element is the file path to the MCD12Q1 file 
            for which the year in the filename is the same as in `modis_file`.
            The second element refers to the (very) next year. 
            If either element is not found in `mcd12q1_files`, it will be None.
    """
    if modis_file is not None:
        if vh_year is not None:
            raise ValueError("Either vh_year or modis_file must be passed, not both")
        meta = meta_from_hdf_filename(modis_file)
        this_year = meta["date"].year
        v, h      = meta["v"], meta["h"]
    elif vh_year is not None:
        v, h, this_year = vh_year
    else: 
        raise ValueError("")

    next_year   = this_year + 1
    hv_tile_str = f"h{h:02d}v{v:02d}"

    # put together the starts of the filenames
    mcd12q1_prefix_this_year = f"MCD12Q1.A{this_year}001.{hv_tile_str}."
    mcd12q1_prefix_next_year = f"MCD12Q1.A{next_year}001.{hv_tile_str}."

    mcd12q1_fname_this_year = [
        f for f in mcd12q1_files 
        if os.path.basename(f).startswith(mcd12q1_prefix_this_year)]
    if len(mcd12q1_fname_this_year) == 0:
        mcd12q1_fname_this_year = None
    else:
        mcd12q1_fname_this_year = mcd12q1_fname_this_year[0]

    mcd12q1_fname_next_year = [
        f for f in mcd12q1_files 
        if os.path.basename(f).startswith(mcd12q1_prefix_next_year)]
    if len(mcd12q1_fname_next_year) == 0:
        mcd12q1_fname_next_year = None
    else:
        mcd12q1_fname_next_year = mcd12q1_fname_next_year[0]

    return mcd12q1_fname_this_year, mcd12q1_fname_next_year

def make_vh_filter(vmin: int, vmax: int, hmin: int, hmax: int
                  ) -> Callable[[str], bool]:
    def f(fp: str) -> bool:
        v = v_from_hdf_filename(fp)
        h = h_from_hdf_filename(fp)
        return (vmin <= v <= vmax) and (hmin <= h <= hmax)
    return f

def get_previous_monthly_hdfpath(query_fpath: str, fpaths: Iterable[str]) -> str:
    """
    Returns the fpath of the MCD64A1 hdf that corresponds to the month before.
    Raises an exception if that fpath can't be found in `fpaths`.
    
    Args:
        query_fpath (str):
            filepath to the hdf of the current month. 
    """
    meta = meta_from_hdf_filename(query_fpath)
    product = meta["product"]
    Y, M = meta["date"].year, meta["date"].month
    if M == 1:
        Y -= 1; M == 12
    else:
        M -= 1
    h, v = meta["h"], meta["v"]
    pattern = f"{Y}.{M:02d}.01/{product}\\.A{Y}[0-9]+\\.h{h:02d}v{v:02d}\\.006\\.[0-9]+\\.hdf$"
    for fp in fpaths:
        if like(fp, pattern):
            return fp
    raise RuntimeError("Couldn't find the fpath to the MCD64A1 file one month "
                       f"before ({query_fpath})")

def h__v__(h: int, v: int) -> str:
    """
    Example: Returns "h11v08" for h=11 and v=8, as in MODIS HDF filenames.
    """
    return f"h{h:02d}v{v:02d}"

def vh_edge_coords(v, h, corner_points_only: bool=False):
    """
    Computes lat lons of all points along the edge of a vh-tile (clockwise).
    Resolution is 1200 steps for each side.

    Args:
        corner_points_only (bool, opt):
            If True, will only return the coordinates of the corners of the 
            vh-tile. This will speed up plotting on maps with sinusoidal
            projection. On other projections this will give false shapes.
            By default False.
    Returns:
        xy (np.ndarray): shape (4797, 2)
            first column: lon
            2nd column:   lat
    """
    n = 1200
    if corner_points_only:
        I = np.r_[0, 0, 1199, 1199, 0]
        J = np.r_[0, 1199, 1199, 0, 0]
    else:
        I = np.concatenate([np.repeat(0, n), 
                            np.arange(1, n), 
                            np.repeat(n-1, (n)-1), 
                            np.arange(n-2, -1, -1)])
        J = np.concatenate([I[n-1:], I[:n-1]])
    v = np.repeat(v, len(I))
    h = np.repeat(h, len(I))
    lat, lon = navigate_inverse(v, h, I, J, res=1)
    xy = np.hstack([lon[:, None], lat[:, None]])
    return xy
