from typing import Callable, Any, List, Iterable, Tuple, Optional, Union, Dict

import sklearn.calibration as skc

# Model generator stuff
def add_calibration_to_modgen(modgen: Callable, method: str, cv=None
                             ) -> Callable:
    def new_modgen():
        model = modgen()
        return skc.CalibratedClassifierCV(model, method=method, cv=cv)
    return new_modgen

def add_calibrated_modgens_to_modgen_dict(modgens: Dict[str,Callable], cv=None
                                         ) -> Dict[str,Callable]:
    new_modgens_dict = dict()
    for k,v in modgens.items():
        new_modgens_dict[k]        = v
        new_modgens_dict[k+"_iso"] = add_calibration_to_modgen(
            v, method="isotonic", cv=cv)
        new_modgens_dict[k+"_sig"] = add_calibration_to_modgen(
            v, method="sigmoid", cv=cv)
    return new_modgens_dict

def materialize(modgens: Dict[str, Callable]) -> Dict[str, Any]:
    models_dict = dict()
    for k,v in modgens.items():
        models_dict[k] = v()
    return models_dict