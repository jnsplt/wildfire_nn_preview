import numpy as np
import pandas as pd


def agg_tail(X: np.ndarray, agg_start: int) -> np.ndarray:
    """
    Aggregates an array from a certain index on using one or more aggregation
    functions, and returns the aggregated part along with the aggregations as
    one array. 

    Parameters
    ----------
    X : np.ndarray : shape (m,n)
        [description]
    agg_start : int
        column index where the tail begins, i.e. from which to start aggregating.
    
    Returns
    -------
    np.ndarray
        [description]
    """
    # temporarily deleted from doc:
    # aggs : Iterable[Callable] : len > 0
    #     functions to call on the part to be aggregated, each with signature
    #     `f(Z, axis, keepdims) -> np.ndarray`
    #     where Z is shape (m,<=n) and Y is shape

    unagg_part = X[:, :agg_start]
    to_be_agg  = X[:, agg_start:]

    # agg_results = [f(X, axis=axis, keepdims=True) for f in aggs]
    agg_min  = np.nanmin(to_be_agg, axis=1, keepdims=True)
    agg_mean = np.nanmean(to_be_agg, axis=1, keepdims=True)
    agg_max  = np.nanmax(to_be_agg, axis=1, keepdims=True)
    
    return np.hstack([unagg_part, agg_min, agg_mean, agg_max])

def append_onehotencoded_lct(Xy: pd.DataFrame) -> None:
    """
    Appends columns `x_lct_1` to `x_lct_17` to Xy, where each column `x_lct_i '
    is #todo bool / 0 or 1 and indicates whether `lct==i`. Appends with side-effects!

    Parameters
    ----------
    Xy : pd.DataFrame
        DataFrame with column `lct`. Thus may also be `y_index` or `lct_index`.
    """
    for i in range(1,18):
        Xy.loc[:, f"x_lct_{i}"] = (Xy["lct"] == i) #* 1
