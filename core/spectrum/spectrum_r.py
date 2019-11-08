from numpy import zeros, float64, complex64, exp, arctan2, pi, angle
from numpy.fft import fft2, fftshift
from numba import jit

from core.functions import r_to_xy_real, r_to_xy_complex


class SpectrumR:
    def __init__(self, **kwargs):
        self.__beam = kwargs['beam']
        self.__intensity_xy = zeros((2 * self.__beam.n_r, 2 * self.__beam.n_r), dtype=float64)
        self.__kerr_phase_xy = zeros((2 * self.__beam.n_r, 2 * self.__beam.n_r), dtype=float64)
        self.__phase_xy = zeros((2 * self.__beam.n_r, 2 * self.__beam.n_r), dtype=float64)

        self.__spectrum = zeros((2 * self.__beam.n_r, 2 * self.__beam.n_r), dtype=complex64)
        self.__spectrum_intensity = zeros((2 * self.__beam.n_r, 2 * self.__beam.n_r), dtype=complex64)

        self.__vortex_phase = self.__initialize_vortex_phase(self.__beam.m,
                                                             2 * self.__beam.r_max,
                                                             2 * self.__beam.n_r,
                                                             self.__beam.dr)

    @property
    def intensity_xy(self):
        return self.__intensity_xy

    @property
    def kerr_phase_xy(self):
        return self.__kerr_phase_xy

    @property
    def phase_xy(self):
        return self.__phase_xy

    @property
    def spectrum_intensity(self):
        return self.__spectrum_intensity

    def __make_fft(self, arr):
        self.__spectrum = fft2(arr)
        self.__spectrum = fftshift(self.__spectrum, axes=(0, 1))

    @staticmethod
    @jit(nopython=True)
    def __initialize_vortex_phase(m, perp_max, n_perp, d_perp):
        vortex_phase = zeros((n_perp, n_perp), dtype=complex64)
        for i in range(n_perp):
            for j in range(n_perp):
                x, y = d_perp * i - 0.5 * perp_max, d_perp * j - 0.5 * perp_max
                vortex_phase[i, j] = exp(1j * m * (arctan2(x, y) + pi))

        return vortex_phase

    def update_data(self):
        # intensity
        self.__intensity_xy = r_to_xy_real(self.__beam._intensity)

        # field
        field_xy = r_to_xy_complex(self.__beam._field)

        # kerr phase
        self.__kerr_phase_xy = angle(field_xy)

        # full phase
        field_xy *= self.__vortex_phase
        self.__phase_xy = angle(field_xy)

        # spectrum
        self.__make_fft(field_xy)
        self.__spectrum_intensity = self.__beam._field_to_intensity(self.__spectrum)

