import typing

from hypothesis import strategies as st

import swiift.model.frac_handlers as fh


class FloatsKWA(typing.TypedDict, total=False):
    allow_nan: bool
    allow_infinity: bool
    allow_subnormal: bool


FloatSt = st.SearchStrategy[float]
float_kw: FloatsKWA = {
    "allow_nan": False,
    "allow_subnormal": False,
}
fracture_handlers = (
    fh.BinaryFracture,
    fh.BinaryStrainFracture,
    fh.MultipleStrainFracture,
)
