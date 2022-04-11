import os
import numpy as np
import pandas as pd
import gdal
from datetime import datetime

from typing import Dict, Any, Iterable, List

import fire.utils.etc as uetc

def get_variable_metadata(gdal_ds: gdal.Dataset, i: int = 1) -> Dict[str, Any]:
    """
    Args:
        i (int): 1-based index of the variable (passed to `GetRasterBand`).

    Returns:
        dictionary with:
            comment (str): 
            forecast_secs (int): 
            forecast_hours (float): 
            datetime (datetime): 
    """
    b = gdal_ds.GetRasterBand(i)

    comment       = b.GetMetadataItem("GRIB_COMMENT")
    short_name    = b.GetMetadataItem("GRIB_SHORT_NAME")
    unit          = b.GetMetadataItem("GRIB_UNIT")
    forecast_secs = b.GetMetadataItem('GRIB_FORECAST_SECONDS')
    forecast_secs = int(uetc.extract(forecast_secs, "[0-9]+"))
    
    dtime = b.GetMetadataItem('GRIB_VALID_TIME')
    dtime = uetc.extract(dtime, "[0-9]+")
    dtime = datetime.utcfromtimestamp(int(dtime))

    # assert dtime.time() == datetime.min.time(), \
    #     "Sth fishy happened: GRIB_VALID_TIME is not at time 00:00."
    # dtime = dtime.date()

    output = {
        "comment": comment,
        "short_name": short_name,
        "unit": unit, 
        "forecast_secs": forecast_secs,
        "forecast_hours": forecast_secs/60/60,
        "datetime": dtime
    }

    return output


def overview(filepath: str, max_bands: int = 1000) -> None:
    """
    Prints an overview of the ERA5 grib file.
    """
    gdal_ds = gdal.Open(filepath)

    #todo get some meta date wrt to the entire file
    # raster shape (first one is enough)
    # crs
    # transform / affine?

    n_bands_to_print = min( (gdal_ds.RasterCount, max_bands) )
    n_digits         = int(np.log10(n_bands_to_print) + 1)

    for i in range(1, max_bands+1): # gdal works with 1-based indices
        b = gdal_ds.GetRasterBand(i)

        if b is None:
            break

        var_meta       = get_variable_metadata(gdal_ds, i)
        formatted_date = var_meta["datetime"].strftime(r"%Y-%m-%d %H:%M")
        var_num_string = f"%{n_digits}d" % i #todo make util function (also for n_digits)
        print(f"{var_num_string}: {formatted_date}, {var_meta['comment']}")


def make_era5_index_from_paths(era5_files: Iterable[str]) -> pd.DataFrame:
    fname_dates = []
    for f in era5_files:
        fname    = os.path.basename(f)
        date_str = uetc.extract(fname, r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
        fname_dates.append( datetime.strptime(date_str, r"%Y-%m-%d") )

    era5_index = pd.DataFrame({
        "fpath": era5_files,
        "fname_date": fname_dates,
        "fname": [os.path.basename(fp) for fp in era5_files],
        "product": "ERA5"
    })

    return era5_index