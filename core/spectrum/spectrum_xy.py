from numpy import zeros, float64, complex64, angle
from numpy.fft import fft2, fftshift
import numpy as np


class SpectrumXY:
    def __init__(self, **kwargs):
        self.__beam = kwargs['beam']
        self.__intensity_xy = zeros((self.__beam.n_x, self.__beam.n_y), dtype=float64)
        self.__phase_xy = zeros((self.__beam.n_x, self.__beam.n_y), dtype=float64)

        self.__spectrum = zeros((self.__beam.n_x, self.__beam.n_y), dtype=complex64)
        self.__spectrum_intensity = zeros((self.__beam.n_x, self.__beam.n_y), dtype=complex64)

    @property
    def intensity_xy(self):
        return self.__intensity_xy

    @property
    def phase_xy(self):
        return self.__phase_xy

    @property
    def spectrum_intensity(self):
        return self.__spectrum_intensity

    def __make_fft(self, arr):
        self.__spectrum = fft2(arr)
        self.__spectrum = fftshift(self.__spectrum, axes=(0, 1))

    def update_data(self):
        # intensity
        self.__intensity_xy = self.__beam._intensity

        field_xy = self.__beam._field

        # phase
        self.__phase_xy = angle(field_xy)

        print('update_data')
        print('max =', np.max(self.__phase_xy))
        print('min =', np.min(self.__phase_xy))

        # spectrum
        self.__make_fft(field_xy)
        self.__spectrum_intensity = self.__beam._field_to_intensity(self.__spectrum)

