#
import numpy as np
import pandas as pd
from datetime import datetime

import warnings

from typing import List, Tuple, Optional, Callable, Union, Dict

# geo stuff
import rasterio as rio # for dataset reading
from rasterio.errors import RasterioIOError
import pyproj # for projection stuff
# from affine import Affine # class for transform matrices

# own stuff
import fire.utils.modis as umod
import fire.utils.io as uio
import fire.utils.geo as ugeo
from fire.utils.etc import ProgressDisplay


def modis__compare_lat_lon_from_nav_and_rasterio_pyproj(
        fpaths: List[str], sds_id: int=0, 
        # raster_id: int=0, 
        ii: np.array=np.array([0,   0,1199,1199,600]), 
        jj: np.array=np.array([0,1199,   0,1199,600]),
        ij_names: List[str]=["ul","ur","ll","lr","center"],
        override_src_crs = None,
        pyproj_Geod_kwargs = {"ellps": "WGS84"}
    ) -> pd.DataFrame:
    """
    """
    all_dfs = []

    not_readable_hdfs: int = 0

    progress = ProgressDisplay(len(fpaths))
    progress.start_timer().print_status()
    for f in fpaths:
        v, h = umod.v_from_hdf_filename(f), umod.h_from_hdf_filename(f)

        sds_path = uio.get_subdataset_path(f, sds_id)
        
        try:
            # open sub/scientific-dataset
            rio_sds = rio.open(sds_path, mode="r")

            # compute coords with rasterio & pyproj
            lons_rp, lats_rp = ugeo.get_coords_for_pixels(
                rio_sds, rows = ii, cols = jj, 
                offset = "center", dst_crs = None, src_crs=override_src_crs)
        except RasterioIOError:
            lons_rp, lats_rp = np.nan, np.nan
            not_readable_hdfs += 1
        
        # compute coords with navigate_inverse defined in MODIS doc
        lats_nav, lons_nav = umod.navigate_inverse(v, h, ii, jj)

        all_dfs.append(pd.DataFrame({
            "i": ii,
            "j": jj,
            "descr": ij_names,
            "lon_rp": lons_rp,
            "lat_rp": lats_rp,
            "lat_nav": lats_nav,
            "lon_nav": lons_nav,
            "v": v,
            "h": h,
            "fname": f
        }))

        progress.update_and_print()

    progress.stop()
    print(f"Number of HDFs that couldn't be read: {not_readable_hdfs}")
    
    df = pd.concat(all_dfs, axis=0)

    # compute distances between different lat lon calculations
    geod = pyproj.Geod(**pyproj_Geod_kwargs) # ellps="WGS84"
    _, _, distance_in_meters = geod.inv(
        lons1 = df["lon_rp"].to_numpy(),  lats1 = df["lat_rp"].to_numpy(), 
        lons2 = df["lon_nav"].to_numpy(), lats2 = df["lat_nav"].to_numpy())
    df.loc[:, "dist_km"] = distance_in_meters/1000

    return df

def modis__navigate_back_and_forth_stays_same() -> bool:
    hs, vs = [1,19,22,6,22,31,35], [8,16,9,3,11,13,10] # chosen arbitrarily
    ii     = [0,600,1199] # recycled as jj

    for res in [1,2,4]:
        for h, v in zip(hs, vs):
            for i in ii:
                j = i
                lat_out, lon_out = umod.navigate_inverse(
                    v=v, h=h, i=i, j=j, res=res, radians=False
                )
                v_out, h_out, i_out, j_out = umod.navigate_forward(
                    lat=lat_out, lon=lon_out, res=res, radians=False
                )
                if v_out!=v or h_out!=h or i_out!=i or j_out!=j:
                    warnings.warn("navigating back and forth failed for "
                                  f"v={v}, h={h}, i={i}, j={j}, res={res}. "
                                  f"Yielded v={v_out}, h={h_out}, i={i_out}, "
                                  f"j={j_out}"
                                 )
                    return False
    return True
