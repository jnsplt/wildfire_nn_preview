import numpy as np
import pandas as pd

import rasterio as rio # for dataset reading
import pyproj # for projection stuff
from affine import Affine # class for transform matrices
from shapely.geometry import Point
import geopandas as gpd
gpd.options.use_pygeos = False # otherwise sjoin fails in country_names_from_lat_lons

import warnings
from datetime import datetime

import gdal
import fire.utils.modis as umod
import fire.utils.era5 as uera
import fire.utils.io as uio
from rasterio.errors import NotGeoreferencedWarning
import os

from typing import List, Tuple, Optional, Iterable, Union, Dict, Any

import fire.utils.etc as uetc

# CONSTANTS
CRS_LATLON = pyproj.CRS.from_epsg(4326)
NATEARTH_MISSING_ISO3 = { # natural earth shapes have some missing country 
                          # codes although the long names work. This is a 
                          # lookup for those missing codes.
    "France": "FRA", # no idea, why this isn't working
    "Norway": "NOR", # ...same here
    "Somaliland": "SOM", 
    "Kosovo": "XKX", # this code is the one that is used in WDI data as well
    "Baikonur": "KAZ", # city in Kazakhstan
    "N. Cyprus": "CYP" # no ISO A3 code for North Cyprus => now it's Cyprus
}


#todo del
def get_coords_for_pixels(*args, **kwargs):
    warnings.warn("Use get_coords_for_rowcols instead", DeprecationWarning)
    return get_coords_for_rowcols(*args, **kwargs)


def get_coords_for_rowcols(dataset: rio.DatasetReader, 
                           rows: Iterable[float], 
                           cols: Iterable[float], 
                           src_crs: Optional[pyproj.crs.CRS] = None,
                           dst_crs: Optional[pyproj.crs.CRS] = None,
                           offset: Union[List[str], str] = "center"
                          ) -> Tuple[List[float], List[float]]:
    """
    Args:
        dataset (rasterio.DatasetReader): must be opened (?) #todo
        rows (numpy.array): indices of the rows of the pixels of interest
        cols (numpy.array): indices of the columns of the pixels of interest
        src_crs (pyproj.CRS): If not None, will be taken instead of the CRS set 
            in the dataset. Defaults to None.
        dst_crs (pyproj.CRS): CRS to project coordinates to. If None, 
            EPSG:4326 (lat lon) will be used. Defaults to None.
        offset (list of str or str): Passed to `rasterio.transform.xy`. One
            of: center, ul, ur, ll, lr.
    
    Returns:
        tuple: two lists of floats, xs and ys. If dst_crs is default, output 
            will be lon, lat. #todo or lat lon??
    """
    warnings.warn("Deprecated function, use rowcols_to_coords instead")
    warnings.warn("Deprecated function, use rowcols_to_coords instead", 
                  DeprecationWarning)

    # sanity check
    assert len(rows) == len(cols), \
        "rows and cols must be lists or arrays of same length"
    
    # set destination CRS (projection) to lat lon, if not given
    if dst_crs is None:
        dst_crs = CRS_LATLON

    # get projection info about dataset
    src_tf:  Affine = dataset.transform
    if src_crs is None:
        src_crs: pyproj.crs.CRS = pyproj.CRS.from_wkt(dataset.crs.to_wkt())
    
    # get coordinates of pixel ijs in src projection
    src_xs, src_ys = rio.transform.xy(src_tf, rows, cols, offset=offset)
    src_xs, src_ys = np.array(src_xs), np.array(src_ys)
    
    # reproject coordinates to destination CRS (projection)
    dst_xs, dst_ys = pyproj.transform(src_crs, dst_crs, src_xs, src_ys, 
                                      errcheck=True, always_xy=True, 
                                      skip_equivalent=True)
    
    return dst_xs, dst_ys

def rowcols_to_coords(rows: Iterable[float], 
                      cols: Iterable[float], 
                      src_tf: Affine, 
                      src_crs: Optional[pyproj.crs.CRS],
                      dst_crs: Optional[pyproj.crs.CRS] = None,
                      offset: Union[List[str], str] = "center"
                     ) -> Tuple[List[float], List[float]]:
    """
    Args:
        rows (numpy.array): indices of the rows of the pixels of interest
        cols (numpy.array): indices of the columns of the pixels of interest
        src_tf (Affine): ... #todo
        src_crs (pyproj.CRS): If not None, will be taken instead of the CRS set 
            in the dataset. Defaults to None.
        dst_crs (pyproj.CRS): CRS to project coordinates to. If None, 
            EPSG:4326 (lat lon) will be used. Defaults to None.
        offset (list of str or str): Passed to `rasterio.transform.xy`. One
            of: center, ul, ur, ll, lr.
    
    Returns:
        tuple: two lists of floats, xs and ys. If dst_crs is default, output 
            will be lon, lat. #todo or lat lon??
    """
    # sanity check
    assert len(rows) == len(cols), \
        "rows and cols must be lists or arrays of same length"
    
    # set destination CRS (projection) to lat lon, if not given
    if dst_crs is None:
        dst_crs = CRS_LATLON
    
    # get coordinates of pixel ijs in src projection
    src_xs, src_ys = rio.transform.xy(src_tf, rows, cols, offset=offset)
    src_xs, src_ys = np.array(src_xs), np.array(src_ys)
    
    # reproject coordinates to destination CRS (projection)
    dst_xs, dst_ys = pyproj.transform(src_crs, dst_crs, src_xs, src_ys, 
                                      errcheck=True, always_xy=True, 
                                      skip_equivalent=True)
    
    return dst_xs, dst_ys


#todo not finished
def get_rowcols_for_coords(xs: Iterable[float], 
                           ys: Iterable[float], 
                           tf: Affine,
                           xy_crs: Optional[pyproj.crs.CRS] = None,
                           tf_crs: Optional[pyproj.crs.CRS] = None,
                           **kwargs
                          ) -> Tuple[Iterable[int], Iterable[int]]:
    """
    Determines the row- and column-ids of pixels in a raster for coordinates.

    Args:
        xs (numpy.array): x-component of coordinate (in EPSG:4326 ("lat-lon"): 
            lon (!)). Must be in the same CRS of the raster to which the 
            transform `tf` belongs, unless `xy_crs` is given. 
        ys (numpy.array): y-component of coordinate 
            (in EPSG:4326 ("lat-lon"): lat (!)).
        tf (Affine): Transform matrix of the raster for which rows and columns
            shall be determined. 
        xy_crs (pyproj.CRS): If None, x and y will be assumed to lie in the same 
            CRS as the raster `tf` originates from. If not None, this CRS and 
            `tf_crs` must be given as well. This way x and y can be (e.g.) in 
            lat-lon, whereas `tf` originates from a sinusoidal raster. Defaults 
            to None. 
        tf_crs (pyproj.CRS): CRS of the raster from which `tf` originates. Not 
            needed and thus ignored if `xy_crs` is None. Defaults to None.
        **kwargs: Keyword arguments passed to `rasterio.transform.rowcol`. 
            Keywords not already used are `op` and `precision`.
    
    Returns:
        tuple: two lists of ints; rows and cols. Could be other datatypes as 
            ints if certain functions are passed to `op` of 
            `rasterio.transform.rowcol` via **kwargs.
    """
    # sanity check
    assert len(xs) == len(ys), \
        "xs and ys must be lists or arrays of same length"

    if xy_crs is not None: # means x and y are in different CRS than the raster 
        if tf_crs is None:
            raise ValueError("If xy_crs is given, tf_crs must be given as well.")
        
        # reproject coordinates to CRS of raster
        transformer = pyproj.Transformer.from_crs(
            xy_crs, tf_crs, always_xy=True, skip_equivalent=True)
        xs, ys = transformer.transform(xs, ys, errcheck=True)
    else:
        if tf_crs is not None:
            warnings.warn("tf_crs is ignored since xy_crs is not given.", 
                          UserWarning)

    return rio.transform.rowcol(tf, xs, ys, **kwargs) # -> rows, cols


def country_names_from_lat_lons(
        lats: Iterable[float], lons: Iterable[float], country_shapes_fpath: str, 
        cols_to_return: Union[str,List[str]]=["NAME_EN","CONTINENT"]
    ) -> Union[pd.DataFrame, pd.Series]:
    """
    
    Args:
        lats, lons (Iterable[float]): 
            (e.g.) lists of latitude and longitude values in degrees.
        country_shapes_fpath (str): 
            filepath to `ne_10m_admin_0_countries.shp` or similar.
        cols_to_return (str or List[str]):
            columns to return, see details.
        
    Details:
        Download shapes file from 
        https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-countries/
        
        Available columns to return:
        'featurecla', 'scalerank', 'LABELRANK', 'SOVEREIGNT', 'SOV_A3',
        'ADM0_DIF', 'LEVEL', 'TYPE', 'ADMIN', 'ADM0_A3', 'GEOU_DIF', 'GEOUNIT',
        'GU_A3', 'SU_DIF', 'SUBUNIT', 'SU_A3', 'BRK_DIFF', 'NAME', 'NAME_LONG',
        'BRK_A3', 'BRK_NAME', 'BRK_GROUP', 'ABBREV', 'POSTAL', 'FORMAL_EN',
        'FORMAL_FR', 'NAME_CIAWF', 'NOTE_ADM0', 'NOTE_BRK', 'NAME_SORT',
        'NAME_ALT', 'MAPCOLOR7', 'MAPCOLOR8', 'MAPCOLOR9', 'MAPCOLOR13',
        'POP_EST', 'POP_RANK', 'GDP_MD_EST', 'POP_YEAR', 'LASTCENSUS',
        'GDP_YEAR', 'ECONOMY', 'INCOME_GRP', 'WIKIPEDIA', 'FIPS_10_', 'ISO_A2',
        'ISO_A3', 'ISO_A3_EH', 'ISO_N3', 'UN_A3', 'WB_A2', 'WB_A3', 'WOE_ID',
        'WOE_ID_EH', 'WOE_NOTE', 'ADM0_A3_IS', 'ADM0_A3_US', 'ADM0_A3_UN',
        'ADM0_A3_WB', 'CONTINENT', 'REGION_UN', 'SUBREGION', 'REGION_WB',
        'NAME_LEN', 'LONG_LEN', 'ABBREV_LEN', 'TINY', 'HOMEPART', 'MIN_ZOOM',
        'MIN_LABEL', 'MAX_LABEL', 'NE_ID', 'WIKIDATAID', 'NAME_AR', 'NAME_BN',
        'NAME_DE', 'NAME_EN', 'NAME_ES', 'NAME_FR', 'NAME_EL', 'NAME_HI',
        'NAME_HU', 'NAME_ID', 'NAME_IT', 'NAME_JA', 'NAME_KO', 'NAME_NL',
        'NAME_PL', 'NAME_PT', 'NAME_RU', 'NAME_SV', 'NAME_TR', 'NAME_VI',
        'NAME_ZH', 'geometry'
    """
    country_boundaries = gpd.read_file(country_shapes_fpath)
    locations = gpd.GeoDataFrame(
        crs='epsg:4326', geometry=[Point(xy) for xy in zip(lons, lats)])
    lookups = gpd.sjoin( # speed depends on order of DFs a lot! 
        country_boundaries, locations, how='right', op='intersects')

    # some country codes are not found and need to be filled in. Fortunately
    # NAME (long country name) usually works. Some fill-ins might not be perfect 
    # but they are good enough for the tasks in this project.
    country_code_missing = lookups["ISO_A3"] == "-99"
    lookups.loc[country_code_missing, "ISO_A3"] = \
        lookups.loc[country_code_missing, "NAME"].replace(NATEARTH_MISSING_ISO3)
    
    return lookups[cols_to_return]

def fill_in_missing_country_codes_in_natearth_data(df: pd.DataFrame, 
                                                   iso3_col: str, 
                                                   long_col: str="NAME"
                                                  ) -> pd.DataFrame:
    """
    Some country codes are not found and need to be filled in. Fortunately
    the long country name in natural earth usually works. Some fill-ins might 
    not be perfect but they are good enough for the tasks in this project.

    REPLACES VALUES IN-PLACE! / SIDE-EFFECTS

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with a column that holds the natural earth long name of 
        countries, and a column where the corresponding ISO-A3 code is to be 
        filled in. (Only fills in a hand-full of countries, those of which the 
        ISO-A3 code is usually missing in naturalearth). `df` is modified
        in-place!
    iso3_col : str
        name of column where ISO-A3 codes should be filled in. Missing ones 
        have to have the value "-99" (str), as always in naturalearth.
    long_col : str
        name of column where the long form of country names is stored. In 
        naturalearth by default "NAME".

    Returns
    -------
        returns df, but only in order to work with pandas.DataFrame.pipe, 
        works in-place
    """
    country_code_is_missing = df[iso3_col] == "-99"
    df.loc[country_code_is_missing, iso3_col] = \
        df.loc[country_code_is_missing, long_col].replace(NATEARTH_MISSING_ISO3)
    return df

#todo
# contains x, y, and CRS
# defaults to lat lon (if crs is None)
class CoordinateArray:
    pass


#todo
class GeoDataset:
    vars: List[str]
    datetimes: List[datetime]

    def __init__(self):
        pass

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}({self.__repr_extra__()})"

    def __repr_extra__(self) -> str:
        return ""

    # @property
    # def transform(self) -> Affine:
    #     raise NotImplementedError

    # @property
    # def bounds(self) -> "???":
    #     raise NotImplementedError

    # @property
    # def crs(self) -> pyproj.crs.CRS:
    #     raise NotImplementedError

    # @property
    # def product(self) -> str:
    #     """
    #     Name of the product from which the dataset comes. E.g. "MOD14A1".
    #     """
    #     raise NotImplementedError

    # @property
    # def identifier(self) -> str:
    #     """
    #     A string that identifies this dataset among all datasets in the 
    #     respective product. E.g. for a MOD14A1 dataset this would be the 
    #     "science dataset" name (e.g. "FireMask"), the region (vh-tile), and the 
    #     start date of the HDF file.
    #     """
    #     raise NotImplementedError

    # @property
    # def all_meta(self) -> Dict[str,Any]:
    #     raise NotImplementedError
    
    # def __repr__(self) -> "???":
    #     raise NotImplementedError


# possible problems:
#   ERA5 hat viele Variablen in einer Datei. In mehrere Datasets aufteilen?
#   Gilt dann genau so für MODIS (FireMask getrennt von MaxFRP usw)
#   zu viele DatasetReader (zu viele Datei Öfnungen? => Performance)
# Was soll ein Dataset beinhalten? 
#   Nur ein Datum? 
#   Nur ein band (ca so wie eine Variable)?
#   Nur ein Subdataset?
#   DEFINITIV nur ein transform, ein bounds, ein CRS
# In welcher Reihenfolge später in GeoDatasetCollection verarbeiten?
#   ERA5 hat riesige files (global), daher lieber alles bzgl. eines Datums machen,
#   was verarbeitet werden muss, und dann erst das nächste Datum laden 
#   (outer loop: date)
#   Bei MODIS tiles sind 8 Tage in einem File. => 8 dates verarbeiten
#   (outer loop: region)



class ModisDataset(GeoDataset):
    def __init__(self, filepath, 
                 var_name_replacements: Optional[Dict[str,str]] = None):

        super(ModisDataset, self).__init__()
        self.fpath      = filepath
        self.product    = umod.product_name_from_hdf_filename(self.fpath)
        self.identifier = os.path.basename(self.fpath)

        self._meta_cache = None # if self.all_meta is called, the output 
                                # will be stored here

        # MODIS specific
        self.vh = (umod.v_from_hdf_filename(self.identifier), 
                   umod.h_from_hdf_filename(self.identifier))

        #todo sanity check
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)

            with rio.open(self.fpath, mode="r") as ds:
                # variables
                self._sds_paths = ds.subdatasets
                self.vars       = [uetc.extract(sds_path, r"[^:]+$") 
                                   for sds_path in self._sds_paths]

                # user defined alternative variable names
                if var_name_replacements is not None:
                    for i,v in enumerate(self.vars):
                        if v in var_name_replacements.keys():
                            self.vars[i] = var_name_replacements[v]

                self._var_to_sds_index: Dict[str,int] = {var: i for i, var in 
                                                         enumerate(self.vars)}
                
                # assume that all sds share the following properties
                with rio.open(self._sds_paths[0], mode="r") as sds:
                    self.transform = sds.transform
                    self.bounds    = sds.bounds
                    self.crs = pyproj.crs.CRS.from_string(sds.crs.to_string())

                    # datetimes
                    dates: Union[str, None] = sds.get_tag_item("Dates")
                    if dates is not None:
                        # one MODIS HDF for several dates, e.g. MOD14A1
                        dates = dates.split()
                    else: 
                        # MODIS HDF for single date, e.g. MCD12Q1
                        dates = [sds.get_tag_item("RANGEBEGINNINGDATE")]
                    self.datetimes = [datetime.strptime(d, r"%Y-%m-%d") 
                                      for d in dates]

                    # rasterios read() takes 1-based indices
                    self._dt_to_raster_index: Dict[datetime, int] = {
                        dt: i+1 for i, dt in enumerate(self.datetimes)}
        
    def get_raster(self, var: str, dt: Optional[datetime] = None) -> np.ndarray:
        i_var = self._var_to_sds_index[var]
        try:
            i_raster = self._dt_to_raster_index[dt] # None will also not be found
        except KeyError:
            i_raster = None
        
        with rio.open(self._sds_paths[i_var], mode="r") as sds:
            if i_raster is None:
                return sds.read()
            else:
                return sds.read(i_raster)

    @property
    def all_meta(self) -> Dict[str,str]:
        if self._meta_cache is None:
            gds = gdal.Open(self.fpath)
            self._meta_cache = gds.GetMetadata_Dict()
        return self._meta_cache
    
    def __repr_extra__(self) -> str:
        return f"filepath = {self.fpath}"
            

class Era5Dataset(GeoDataset):
    """
    Args:
        override_var_names (list of str): Variable names to use instead of the
            names stored in meta data to each variable. Especially useful, when 
            ERA5 grib file contains variable names "undefined [-]". This list is
            recycled, e.g. when there are 3 variables but 6 bands (due to 2 
            datetimes), the variable name used for the 4th band will be the 
            1st name in `override_var_names`.
    """
    def __init__(self, filepath, 
                 var_name_replacements: Optional[Dict[str,str]] = None,
                 override_var_names: Optional[List[str]] = None):
        super(Era5Dataset, self).__init__()
        self.fpath      = filepath
        self.product    = "ERA5" #todo ERA5-Land or ERA5 ?
        self.identifier = os.path.basename(self.fpath) #todo

        ds = gdal.Open(self.fpath)
        
        # variables
        self.vars = set()
        self.datetimes = set()

        # (var, datetime) -> band_index
        self._var_date_to_band_index: Dict[Tuple[str, datetime], int] = dict()
        # band_index -> var_meta
        self._band_index_to_meta: Dict[int, Dict] = dict()

        if override_var_names is not None:
            override_var_names = uetc.recycle_list_to_len(override_var_names, 
                                                          uio.count_bands(ds))

        band_index = 1 # 1-based index
        while(True):
            # test if a band with this index exists...
            if ds.GetRasterBand(band_index) is None:
                break
            
            # otherwise...
            band_meta = uera.get_variable_metadata(ds, band_index)
            self._band_index_to_meta[band_index] = band_meta

            if override_var_names is None:
                var_name = band_meta["comment"]
            else:
                var_name = override_var_names[band_index-1]
            
            if var_name_replacements is not None:
                if var_name in var_name_replacements.keys():
                    var_name = var_name_replacements[var_name]

            self.vars |= {var_name}

            dt = band_meta["datetime"]
            self.datetimes |= {dt}

            # store band_index for this var and dt
            self._var_date_to_band_index[ (var_name, dt) ] = band_index

            band_index += 1

        self.vars = list(self.vars)
        self.datetimes = list(self.datetimes)

        self.transform = Affine.from_gdal(*ds.GetGeoTransform())
        self.bounds = None #todo with rio?
        self.crs = pyproj.crs.CRS.from_string(ds.GetProjection())

    def get_raster(self, var: str, dt: datetime) -> np.ndarray:
        band_index = self._var_date_to_band_index[var, dt]
        gds = gdal.Open(self.fpath)
        # single line method chaining would crash the kernel
        # see https://github.com/OSGeo/gdal/issues/2684
        # or here https://gdal.org/api/python_gotchas.html 
        # this will not change in future versions (historic behaviour)
        return gds.GetRasterBand(band_index).ReadAsArray()


class GPW4Dataset(GeoDataset):
    """
    For .tif files as downloaded from here:
    https://sedac.ciesin.columbia.edu/data/set/gpw-v4-population-density-rev11/data-download
    
    Args:
        cache_raster (bool, opt.):
            If True, always the most recently loaded raster (via .get_raster) 
            will be stored in-memory and output if the same arguments are
            passed to .get_raster again.    
    """
    def __init__(self, filepath: str, year=None, cache_raster: bool=False):
        if not filepath.endswith(".tif"):
            raise ValueError("file must be a .tif file")
        self.fpath = filepath
        
        if year is None: # then infer from filename
            self.year = int(uetc.extract(os.path.basename(self.fpath), 
                                         r"[12][0-9]{3}"))
        
        rds = rio.open(self.fpath)
        
        assert rds.count == 1, ( 
            f"File has unexpected number of layers (expected 1, got {rds.count}). "
            "See docstring for expected file type.")
        
        self.transform = rds.transform
        self.bounds    = rds.bounds
        self.crs       = pyproj.crs.CRS.from_epsg(rds.crs.to_epsg())
        
        self.vars      = ["gpw4"]
        self.datetimes = [datetime(self.year, 1, 1)]
        self.product   = "GPW4"
        self.shape     = (rds.height, rds.width)
        self.size      = self.shape[0] * self.shape[1]
        rds.close()

        self.cache_raster  = cache_raster
        self._raster_cache = None

    def get_raster(self, var: Optional[str]=None, dt: Optional[datetime]=None
                  ) -> np.ndarray:
        if var is not None and var not in self.vars:
            raise KeyError(f"variable name {var} not in self.vars, just pass None")
        if dt is not None and dt not in self.datetimes:
            raise KeyError(f"datetime {dt} not in self.datetimes, just pass None")

        if (self._raster_cache is not None and self._raster_cache[0] == (var, dt)):
            return self._raster_cache[1]
        else:
            rds     = rio.open(self.fpath)
            raster  = rds.read()[0]
            nan_val = rds.nodata
            rds.close()
            nan_mask = np.isclose(raster, nan_val)
            raster[nan_mask] = np.nan

            if self.cache_raster:
                self._raster_cache = ((var,dt), raster)
            return raster
    
    def __repr_extra__(self) -> str:
        return f"filepath = {self.fpath}"


class GRIP4Dataset(GeoDataset):
    """
    Details:
        GRIP4 dataset can be downloaded here:
        https://www.globio.info/download-grip-dataset
    """
    def __init__(self, fpath: str, cache_raster: bool=True):
        if not fpath.endswith(".asc"):
            raise ValueError("file must be a .asc file")
        self.fpath = fpath

        with rio.open(self.fpath) as rds:
            assert rds.count == 1, (
                "File has unexpected number of layers (expected 1, got "
                f"{rds.count}). See docstring for expected file type.")
            self.transform = rds.transform
            self.bounds    = rds.bounds
            self.crs       = CRS_LATLON
            self.shape     = (rds.height, rds.width)
            self.size      = self.shape[0] * self.shape[1]
        self.cache_raster  = cache_raster
        self._raster_cache = None

    def get_raster(self) -> np.ndarray:
        if self._raster_cache is not None:
            return self._raster_cache
        else:
            with rio.open(self.fpath) as rds:
                raster  = rds.read()[0].astype(float)
                nan_val = rds.nodata
            nan_mask = np.isclose(raster, nan_val)
            raster[nan_mask] = np.nan

        if self.cache_raster:
            self._raster_cache = raster
        return raster
    
    def __repr_extra__(self) -> str:
        return f"filepath = {self.fpath}"


def get_values_from_geodataset(
    ds: GeoDataset, 
    var_dtimes: Optional[Iterable[Tuple[str, datetime]]] = None,
    xys: Optional[Tuple[Iterable[float], Iterable[float]]] = None,
    rowcols: Optional[Tuple[Iterable[int], Iterable[int]]] = None,
    nan_vals: Iterable[Any] = [], 
    xy_crs: Optional[pyproj.crs.CRS] = None,
    var_name_prefix = "", var_name_postfix = ""
) -> pd.DataFrame:
    """
    Args:
        xy_crs: If None, xs and ys are assumend to be in lon/lat. Ignored if 
            rows and cols is given.
    """
    # sanity check of inputs
    if (xys is None) == (rowcols is None):
        raise ValueError("Either xys OR rowcols must be given, not both")
    if (var_dtimes is not None) and (len(var_dtimes) == 0):
        raise ValueError("var_dtimes is empty")
        
    # xy -> rowcol
    if rowcols is None:
        xs, ys = xys
        xs, ys = np.array(xs), np.array(ys)
        if xy_crs is None:
            xy_crs = CRS_LATLON
        rows, cols = get_rowcols_for_coords(
            *xys, ds.transform, xy_crs=xy_crs, tf_crs=ds.crs)
    else:
        rows, cols = rowcols
        rows, cols = np.array(rows), np.array(cols)
        
    # extract values
    var_date_wise_dfs = []
    var_dtimes_passed = var_dtimes is not None
    if not var_dtimes_passed:
        var_dtimes = [None] # ugly code but works to run code in loop just once
    for vd in var_dtimes:
        # get raster
        if vd is None:
            raster = ds.get_raster()
        else:
            var, dt = vd
            raster = ds.get_raster(var, dt)
        raster = uetc.replace_by_nan(raster, nan_vals)
        
        # read values
        values = raster[rows, cols]

        # build dataframe
        if xys is None: # rowcols were passed
            df = pd.DataFrame({"row": rows, "col": cols, "value": values})
        else: # xys were passed
            df = pd.DataFrame({"x": xs, "y": ys, "value": values})
        if var_dtimes_passed:
            var_name = var_name_prefix + var + var_name_postfix
            df.loc[:,"var"]      = var_name
            df.loc[:,"datetime"] = dt
        var_date_wise_dfs.append(df)

    return pd.concat(var_date_wise_dfs, axis=0)

def get_transform(fp: str) -> Affine:
    with rio.open(fp) as rsrc:
        tf = rsrc.transform
    return tf

def lon180_to_lon360(lons):
    """
    Converts longitudes that range from -180 to 180, to longitudes that range
    from 0 to 360.
    """
    lons180 = np.array(lons)
    lons360 = lons180.copy()
    lons180_smaller_0 = lons180 < 0
    lons360[lons180_smaller_0] = 360+lons180[lons180_smaller_0]
    return lons360
