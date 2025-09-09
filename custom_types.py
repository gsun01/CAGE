from __future__ import annotations
__all__ = ["Polarization", "Family", "Sign", "Triple", "TripleC", "TripleV", 
           "c128", "f64", "NDArray", "ArrayLike", "asarr_f64", "asarr_c128"]

from typing import Callable, Literal, Tuple
from numpy.typing import NDArray, ArrayLike
import numpy as np

# custom types
Polarization = Literal["+", "x", "plus", "cross"]
Family = Literal["TE", "TM", "te", "tm", "Te", "Tm"]
Sign = Literal["+", "-"]
TripleC = Tuple[Callable[[ArrayLike, ArrayLike, ArrayLike], NDArray],
               Callable[[ArrayLike, ArrayLike, ArrayLike], NDArray],
               Callable[[ArrayLike, ArrayLike, ArrayLike], NDArray]]
TripleV = Tuple[NDArray, NDArray, NDArray]
Triple = TripleC | TripleV

# type aliases
c128 = np.complex128
f64 = np.float64
NDArray, ArrayLike = NDArray, ArrayLike

def asarr_f64(x):
    return np.asarray(x, dtype=f64)

def asarr_c128(x):
    return np.asarray(x, dtype=c128)