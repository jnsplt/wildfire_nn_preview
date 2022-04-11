from typing import Union
import warnings

try:
    import pyproj
except ModuleNotFoundError as e:
    warnings.warn("Exception caught during import: " + str(e), UserWarning)

from datetime import datetime


def to_pyproj_crs(crs: Union[str, "pyproj.CRS"]) -> "pyproj.CRS":
    if type(crs) is str:
        return pyproj.CRS.from_string(crs)
    elif isinstance(crs, pyproj.CRS):
        # already is of correct type
        return crs
    else:
        raise TypeError("crs must be either of type str or pyproj.CRS")

def to_datetime(date: Union[str, datetime]) -> datetime:
    """
    Converts an input to a `datetime` object, if it's not already one.

    Args:
        date (Union[str, datetime]): String of format `%Y-%m-%d` or 
            `datetime` object. If the latter, it will merely be returned 
            as is.

    Raises:
        TypeError: If the input is neither of type `str` nor `datetime`.

    Returns:
        datetime: [description]
    """
    if type(date) is str:
        return datetime.strptime(date, r"%Y-%m-%d")
    elif isinstance(date, datetime):
        # already is of correct type
        return date
    else:
        raise TypeError("date must be either of type str or datetime")
