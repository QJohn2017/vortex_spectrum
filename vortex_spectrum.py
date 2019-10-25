from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import numpy as np
from numpy import zeros, complex64, float64, exp, arctan2, pi, sqrt, log10, where
from numpy.fft import fft2, fftshift
from numba import jit


class FFTVortex:
    def __init__(self, **kwargs):
        self.__n_perp = kwargs['n_perp']
        self.__perp_max = kwargs['perp_max']
        self.__d_perp = self.__perp_max / self.__n_perp

        self.__m = kwargs['m']
        self.__r_0 = kwargs['r_0']

        self.__kerr_coeff = kwargs['kerr_coeff']

        self.__arr = zeros((self.__n_perp, self.__n_perp), dtype=complex64)
        self.__arr_norm = zeros((self.__n_perp, self.__n_perp), dtype=float64)
        self.__arr_norm_cropped = None

        self.__spectrum = zeros((self.__n_perp, self.__n_perp), dtype=complex64)
        self.__spectrum_norm = zeros((self.__n_perp, self.__n_perp), dtype=float64)
        self.__spectrum_norm_cropped = None

    @staticmethod
    @jit(nopython=True)
    def __initialize_arr(arr, perp_max, d_perp, r_0, m, kerr_coeff):
        N, M = arr.shape[0], arr.shape[1]
        for i in range(N):
            for j in range(M):
                x, y = d_perp * i - 0.5 * perp_max, d_perp * j - 0.5 * perp_max
                r = sqrt(x**2 + y**2)
                arr[i, j] = (r / r_0)**abs(m) * exp(-0.5 * (r / r_0)**2) * \
                            exp(1j * m * (arctan2(x, y) + pi)) * \
                            exp(1j * kerr_coeff * r**2)

    @staticmethod
    @jit(nopython=True)
    def __normalize(arr):
        N, M = arr.shape[0], arr.shape[1]
        arr_norm = zeros((N, M), dtype=float64)
        for i in range(N):
            for j in range(M):
                arr_norm[i, j] = arr[i, j].real**2 + arr[i, j].imag**2

        return arr_norm

    def __make_fft(self):
        self.__spectrum = fft2(self.__arr)
        self.__spectrum = fftshift(self.__spectrum, axes=(0, 1))

    def __crop_arr(self, arr, remaining_central_part_coeff):
        """
        :param remaining_central_part_coeff: 0 -> no points, 0.5 -> central half number of points, 1.0 -> all points
        :return:
        """

        if remaining_central_part_coeff < 0 or remaining_central_part_coeff > 1:
            raise Exception('Wrong remaining_central part_coeff!')

        N = arr.shape[0]
        delta = int(remaining_central_part_coeff / 2 * N)
        i_min, i_max = N // 2 - delta, N // 2 + delta

        return arr[i_min:i_max, i_min:i_max]

    @staticmethod
    def __log_arr(arr):
        return where(arr < 0.1, -1, log10(arr))

    @staticmethod
    def __log_spectrum(arr):
        return log10(arr / np.max(arr))

    def __plot(self):

        fig = plt.figure(figsize=(15, 10), constrained_layout=True)
        spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
        ax1 = fig.add_subplot(spec[0, 0])
        ax2 = fig.add_subplot(spec[0, 1])

        ax1.set_aspect('equal')
        ax2.set_aspect('equal')

        ax1.set_title('$\mathbf{I(x, y)}$')
        ax2.set_title('$\mathbf{S(k_x, k_y)}$')

        arr_for_plot = self.__log_arr(self.__crop_arr(self.__arr_norm, 0.2))
        spectrum_for_plot = self.__log_arr(self.__crop_arr(self.__arr_norm, 0.2))

        ax1.contourf(arr_for_plot, cmap=cm.jet, levels=100)
        ax2.contourf(spectrum_for_plot, cmap=cm.gray, levels=100)

        plt.savefig('fft_vortex.png', bbox_inches='tight')
        plt.close()

    def process(self):
        self.__initialize_arr(self.__arr, self.__perp_max, self.__d_perp, self.__r_0, self.__m, self.__kerr_coeff)
        self.__arr_norm = self.__normalize(self.__arr)

        self.__make_fft()
        self.__spectrum_norm = self.__normalize(self.__spectrum)

        self.__plot()


fft_vortex = FFTVortex(n_perp=1024,
                       perp_max=1000 * 10**-6,
                       m=1,
                       r_0=15 * 10**-6,
                       kerr_coeff=10**5)

fft_vortex.process()