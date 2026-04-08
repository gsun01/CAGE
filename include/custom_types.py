from __future__ import annotations
__all__ = ["Polarization", "Family", "Sign", "TripleC", "TripleV",
            "NDArray", "ArrayLike", "f64", "c128", "asarr_f64", "asarr_c128"]

from typing import Callable, Literal, Tuple
from numpy.typing import NDArray, ArrayLike
import numpy as np

# custom types
Polarization = Literal["+", "x", "plus", "cross"]
Family = Literal["TE", "TM"]
Sign = Literal["+", "-"]
TripleC = Tuple[Callable[[ArrayLike, ArrayLike, ArrayLike], NDArray],
               Callable[[ArrayLike, ArrayLike, ArrayLike], NDArray],
               Callable[[ArrayLike, ArrayLike, ArrayLike], NDArray]]
TripleV = Tuple[NDArray, NDArray, NDArray]

# type aliases
NDArray, ArrayLike = NDArray, ArrayLike
f64 = np.float64 | float
c128 = np.complex128 | complex

def asarr_f64(x):
    return np.asarray(x, dtype=np.float64)

def asarr_c128(x):
    return np.asarray(x, dtype=np.complex128)