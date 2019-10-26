from numpy import zeros, complex64, exp, arctan2, pi
from numba import jit

from .functions import r_to_xy_real, r_to_xy_complex


class Spectrum:
    def __init__(self, **kwargs):
        self.__beam = kwargs['beam']
        self.__intensity_xy = zeros((2 * self.__beam.n_r, 2 * self.__beam.n_r), dtype=complex64)
        self.__spectrum = zeros((2 * self.__beam.n_r, 2 * self.__beam.n_r), dtype=complex64)

        self.__vortex_phase = self.__initialize_vortex_phase(self.__vortex_phase,
                                                             self.__beam.m,
                                                             2 * self.__beam.r_max,
                                                             2 * self.__beam.n_r,
                                                             self.__beam.dr)

    @staticmethod
    @jit(nopython=True)
    def __initialize_vortex_phase(vortex_phase, m, perp_max, n_perp, d_perp):
        for i in range(n_perp):
            for j in range(n_perp):
                x, y = d_perp * i - 0.5 * perp_max, d_perp * j - 0.5 * perp_max
                vortex_phase[i, j] = exp(1j * m * (arctan2(x, y) + pi))

        return vortex_phase

    def update_intensity(self):
        self.__intensity_xy = r_to_xy_real(self.__beam._intensity)

    def update_spectrum(self):
        field_xy = r_to_xy_complex(self.__beam._field)
        field_xy *= self.__vortex_phase
