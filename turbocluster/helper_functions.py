import numpy as np
import cupy as cp


def volume_integral(snap, qt, indices):
    if isinstance(qt, str):
        qt = snap[qt]

    # using paicos omp-vectorized routines
    a = snap.get_sum_of_array((qt * snap["0_Volume"])[indices])

    return a


def volume_average(snap, qt, indices):
    if isinstance(qt, str):
        qt = snap[qt]

    # using paicos omp-vectorized routines
    a = snap.get_sum_of_array((qt * snap["0_Volume"])[indices])
    vol = snap.get_sum_of_array((snap["0_Volume"])[indices])

    return a / vol


def extract_turbulent_var(
    snap,
    sf,
    variable,
    filter_length,
    weight,
    filter_type="gaussian",
    iterative=False,
    selection=None,
):

    filt_var = sf.filter_variable(
        variable,
        filter_length,
        weight=weight,
        filter_type=filter_type,
        iterative=iterative,
        selection=selection,
    )

    print(
        "min/max/avg occupancy cartesian tiling %d / %d / %.2f"
        % (
            sf.tile.particles_per_tile.min(),
            sf.tile.particles_per_tile.max(),
            cp.mean(sf.tile.particles_per_tile),
        )
    )
    if isinstance(variable, str):
        variable = snap[variable]

    smoothVar = np.zeros_like(variable)
    turbVar = np.zeros_like(variable)

    smoothVar[sf.index] = filt_var
    turbVar[sf.index] = variable[sf.index] - filt_var
    turbVar[~sf.indicesFirstPass * sf.index] = (
        0  # this is for the thin buffer, which we set to zero
    )

    return smoothVar, turbVar
