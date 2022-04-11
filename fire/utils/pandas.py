import pandas as pd
import numpy as np
import sys # getsizeof
from datetime import timedelta
import calendar

from typing import List, Tuple, Union, Iterable, Optional

import fire.utils.etc as uetc


def carry_dates_on(df: pd.DataFrame, days: int, src_date_col: str, 
                   dst_date_col: Optional[str] = None,
                   forward: bool = True
                  ) -> pd.DataFrame:
    """
    Copies all rows while incrementing (forward) or decrementing (backward) 
    the date in a given column for each copy.

    Args:
        dst_date_col (str (opt)): If not None, the in- or de-cremented dates 
            will be written to a column with this name. If None, these dates 
            are written to the column `src_date_col`. Defaults to None.

    Example:
        `df`
        foo          date
        -----------------
        bar    2020-01-01
        buzz   2020-01-03

        `carry_dates_on(df, days=3, src_date_col="date")`
        foo          date
        -----------------
        bar    2020-01-01
        bar    2020-01-02
        bar    2020-01-03
        buzz   2020-01-03
        buzz   2020-01-04
        buzz   2020-01-05
    """
    if dst_date_col is None:
        dst_date_col = src_date_col
    
    copies_with_modified_dates = []
    for i in range(days):
        td = timedelta(days=i) if forward else -timedelta(days=i)
        copies_with_modified_dates.append(
            df.copy().assign(**{
                dst_date_col : lambda df: df[src_date_col] + td
            })
        )
    return pd.concat(copies_with_modified_dates)

def categorize(df: pd.DataFrame, exclude: Iterable[str] = [], 
               max_ratio: float = .75, verbose: bool = False) -> pd.DataFrame:
    """
    Converts columns of a pandas DataFrame to dtype "category" iff this leads 
    to a reduction of memory-usage.
    
    Args:
        max_ratio (float): A column is only converted to category if the 
            size as type category is less than `max_ratio` times the original
            size. If you just want to have all columns converted, pass np.inf.
    """
    df = df.copy()
    for col in df.columns:
        if col in exclude:
            if verbose:
                print(f"{col}: skipped")
        else:
            col_categorized  = df[col].astype("category")
            orig_size = df[col].nbytes
            cat_size  = col_categorized.nbytes
            significant_reduction = cat_size < (max_ratio*orig_size)
            
            if verbose:
                orig_mb, cat_mb = orig_size/(1024**2), (cat_size/(1024**2))
                print(f"{col}: {np.round(orig_mb, 2)} MB originally, "
                      f"{np.round(cat_mb, 2)} MB as category. "
                      f"Converted? {significant_reduction}")
                      
            if significant_reduction:
                df[col] = col_categorized
    return df

def implode(df: "pandas.DataFrame", 
            groupby: Union[str, List[str]], 
            return_length_1_lists: bool=True
           ) -> "pandas.DataFrame":
    """
    Groups the dataframe and wraps up the columns in each group to lists. 
    The number of columns stays the same, whereas the number of rows is reduced 
    to the number of groups. For an illustration see the example at the end of
    this docstring.
    
    Meant as counterpart to function pandas.DataFrame.explode.
    
    Args:
        groupby: (str or list of str)
            columns to group by (as passed to 
            pandas.DataFrame.groupby)
        return_length_1_lists: (bool)
            Whether to keep lists of length 1 (True) or to 
            turn them into the only scalar value they hold (False). 
    Returns:
        pandas.DataFrame
        
    Example:
        colA  colB
           a     1
           b     2
           a     3
           
        grouped by colA, becomes...
        colA   colB
           a  [1,3]
           b    [2]
    """
    if type(groupby) == str:
        groupy = [groupby] # for convenience
    
    grouped      = df.groupby(groupby)
    implode_cols = [col for col in df.columns if col not in groupby]
    
    imploded_series = []
    for col in implode_cols:
        ser = (
            grouped
            .apply(lambda grp: list(grp[col]))
            .rename(col)
        )
        if not return_length_1_lists:
            ser = ser.apply(lambda x: x if len(x) > 1 else x[0])
        imploded_series.append(ser)
        
    return pd.concat(imploded_series, axis=1).reset_index()

def notna_ratio(x: pd.Series) -> float:
    """
    The ratio of values in a pandas series that are not NaN.
    """
    return x.notna().sum()/len(x)

def isna_ratio(x: pd.Series) -> float:
    """
    The ratio of values in a pandas series that are NaN.
    """
    return x.isna().sum()/len(x)

def flatten_multilevel_columns(df: pd.DataFrame, sep: str="_", inplace=False
                              ) -> pd.DataFrame:
    new_cols = [sep.join([l for l in levels if l != ""]) 
                for levels in df.columns]
    if not inplace:
        df = df.copy()
    df.columns = new_cols
    return df

def interval_mids(intervals: Iterable[pd.Interval]) -> np.ndarray:
    return np.array([itv.mid for itv in intervals])


# also: checkout pd.MultiIndex.from_product()
# & https://stackoverflow.com/questions/37003100/pandas-groupby-for-zero-values
def cartesian_product(df1, df2=None, 
                      preserve_cols: Union[str, List[str]]=None
                     ) -> pd.DataFrame:
    """
    Does a cartesian product join on df1 itself or df1 and df2.
    
    Args:
        df1 (DataFrame): 
        df2 (DataFrame (opt)): 
        preserve_cols (list of str (opt)): Names of the columns which are 
            in both df1 and df2, which should be kept as is. 
    """
    df1 = df1.copy()
    
    if df2 is None:
        df2 = df1
    else:
        df2 = df2.copy()
    
    # generate a random name for the column that will be used to perform the 
    # join.
    temp_col_name = None
    while ((temp_col_name is None) 
           or (temp_col_name in df1.columns) 
           or (temp_col_name in df2.columns)):
        temp_col_name = uetc.random_string()
    
    # set all values of the temp. col. to 0
    # This way, when joined on this col., all rows of df1 are joined to
    # all of df2.
    df1.loc[:,temp_col_name] = 0
    if df2 is not None: # redundant if df1 is df2
        df2.loc[:,temp_col_name] = 0 
        
    # preserve columns
    if preserve_cols is not None:
        if type(preserve_cols) is str:
            preserve_cols = [preserve_cols]
        join_columns = preserve_cols + [temp_col_name]
    else:
        join_columns = [temp_col_name]
    
    joined = pd.merge(
        df1, df2, on=join_columns, how="outer")
    
    return joined.drop(columns=temp_col_name)

def dt_series_is_every_day_in_one_year(dts: pd.Series) -> bool:
    """
    Checks if a pandas.Series of type datetime is a sorted series of all days 
    of one year (and not more), without duplicates.
    """
    if not hasattr(dts, "dt"):
        return TypeError("dts must be of type datetime (i.e. have a datetime "
                         "accessor `dts.dt`")

    years = np.unique(dts.dt.year)
    if len(years) != 1:
        return False
    
    year  = years.item()
    ndays = 366 if calendar.isleap(year) else 365
    
    actual_days_of_year = dts.dt.dayofyear.values
    days_of_year_as_they_should_be = np.arange(1,ndays+1)
    every_day_is_in_there = np.array_equal(
        actual_days_of_year, days_of_year_as_they_should_be)
    return every_day_is_in_there

def nrows_in_hdf(fp: str, key: str) -> int:
    store = pd.HDFStore(fp)
    nrows = store.get_storer(key).nrows
    store.close()
    return nrows

def sample_with_max_diversity(df, frac, random_state=None):
    """
    Samples rows from a dataframe but keeps all samples in as often as frac 
    allows.
    E.g. if frac=2.3, `df` will be concatenated twice and then concatenated with 
    30% of the rows sampled without replacement. Finally, all rows will be 
    shuffled.
    """
    if frac <= 1:
        return df.sample(frac=frac, 
                         replace=False, 
                         random_state=random_state)
    else:
        sampled = [df] * int(frac)
        sampled.append(df.sample(frac=frac-int(frac), 
                                 replace=False, 
                                 random_state=random_state))
        sampled = pd.concat(sampled).sample(frac=1, 
                                            replace=False, 
                                            random_state=random_state) 
        return sampled
