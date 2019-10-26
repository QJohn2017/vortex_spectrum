from abc import ABCMeta, abstractmethod
from numpy import conj, zeros, complex64
from numba import jit


class DiffractionExecutor(metaclass=ABCMeta):
    """
    Abstract class for diffraction object.
    The class takes on the input in the constructor a beam object, which contains all the necessary beam parameters
    for further calculations.

    Depending on the type of beam diffraction is implemented in different ways. To simulate diffraction in
    a 2-dimensional beam and a 3-dimensional beam in the axisymmetric approximation, a sweep was used;
    to simulate the diffraction of a non-axisymmetric 3-dimensional beam, we used fast Fourier transform with pyfftw
    in several threads.

    """
    def __init__(self, **kwargs):
        self._beam = kwargs['beam']

    @abstractmethod
    def info(self):
        """DiffractionExecutor type"""

    @abstractmethod
    def process_diffraction(self, dz):
        """Process_diffraction"""


class SweepDiffractionExecutorR(DiffractionExecutor):
    """
    Class for modeling the diffraction of a 3-dimensional beam in axisymmetric approximation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # sweep coefficients and arrays

        self.__c1 = 1.0 / (2.0 * self._beam.dr ** 2)
        self.__c2 = 1.0 / (4.0 * self._beam.dr)
        self.__c3 = 2j * self._beam.medium.k_0

        self.__alpha = zeros(shape=(self._beam.n_r,), dtype=complex64)
        self.__beta = zeros(shape=(self._beam.n_r,), dtype=complex64)
        self.__gamma = zeros(shape=(self._beam.n_r,), dtype=complex64)
        self.__vx = zeros(shape=(self._beam.n_r,), dtype=complex64)  # array responsible for accounting topological
                                                                     # charge

        for i in range(1, self._beam.n_r - 1):
            self.__alpha[i] = self.__c1 + self.__c2 / self._beam.rs[i]
            self.__gamma[i] = self.__c1 - self.__c2 / self._beam.rs[i]
            self.__vx[i] = (self._beam.m / self._beam.rs[i]) ** 2  # topological charge accounting

        self.__kappa_left, self.__mu_left, self.__kappa_right, self.__mu_right = \
            1.0, 0.0, 0.0, 0.0

        self.__delta = zeros(shape=(self._beam.n_r,), dtype=complex64)
        self.__xi = zeros(shape=(self._beam.n_r,), dtype=complex64)
        self.__eta = zeros(shape=(self._beam.n_r,), dtype=complex64)

    @property
    def info(self):
        return 'sweep_diffraction_executor_r'

    @staticmethod
    @jit(nopython=True)
    def __fast_process(field, n_r, dz, c1, c3, alpha, beta, gamma, delta, xi, eta, vx,
                     kappa_left, mu_left, kappa_right, mu_right):

        # left boundary condition
        xi[1], eta[1] = kappa_left, mu_left

        # forward
        for i in range(1, n_r - 1):
            beta[i] = 2.0 * c1 + c3 / dz + vx[i]
            delta[i] = alpha[i] * field[i + 1] - \
                       (conj(beta[i]) - vx[i]) * field[i] + \
                       gamma[i] * field[i - 1]
            xi[i + 1] = alpha[i] / (beta[i] - gamma[i] * xi[i])
            eta[i + 1] = (delta[i] + gamma[i] * eta[i]) / \
                         (beta[i] - gamma[i] * xi[i])

        # right boundary condition
        field[n_r - 1] = (mu_right + kappa_right * eta[n_r - 1]) / \
                         (1.0 - kappa_right * xi[n_r - 1])

        # backward
        for j in range(n_r - 1, 0, -1):
            field[j - 1] = xi[j] * field[j] + eta[j]

        return field

    def process_diffraction(self, dz):
        """
        :param dz: current step along evolutionary coordinate z

        :return: None
        """
        self._beam._field = self.__fast_process(self._beam._field, self._beam.n_r, dz, self.__c1,
                                                self.__c3, self.__alpha, self.__beta, self.__gamma, self.__delta,
                                                self.__xi, self.__eta, self.__vx, self.__kappa_left, self.__mu_left,
                                                self.__kappa_right, self.__mu_right)

