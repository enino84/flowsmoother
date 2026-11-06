import numpy as np
import pytest

from flowsmoother.grid import Band


def toy_band(nlat=20, nlon=16, r0=3, r1=14):
    lat = 90.0 - np.arange(nlat) * 180.0 / nlat
    return Band(nlat, nlon, r0, r1, lat)


@pytest.fixture
def band():
    return toy_band()


@pytest.fixture
def tiny():
    """Band small enough for dense references: 3 x 5 cells, joint size 90."""
    return toy_band(nlat=9, nlon=5, r0=3, r1=5)
