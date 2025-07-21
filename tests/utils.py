import typing

from hypothesis import strategies as st


class FloatsKWA(typing.TypedDict, total=False):
    allow_nan: bool
    allow_infinity: bool
    allow_subnormal: bool


FloatSt = st.SearchStrategy[float]
float_kw: FloatsKWA = {
    "allow_nan": False,
    "allow_subnormal": False,
}
