import numpy as np
from abc import ABC
from customexceptions import convergeError

ITERATION_LIMIT = 10000


class abstract(ABC):
    def __init__(self):
        """initialization for a class"""
        self._mtr = None
        self._sol = None

    @staticmethod
    def _generator_random(n) -> np.array:
        return np.random.randint(-20, 21, [n, n]).astype("float")

    def converge_check(self):
        diag = np.diag(np.abs(self._mtr))

        off_diag = np.sum(np.abs(self._mtr), axis=1) - diag

        if np.all(diag > off_diag):
            pass
        else:
            raise convergeError

    @property
    def get_matrix(self) -> np.ndarray:
        return self._mtr

    @get_matrix.getter
    def get_matrix(self) -> np.ndarray:
        return self._mtr

    def _find_b(self):
        self._b = self._mtr @ self._sol
        return self._b

    @property
    def sol(self):
        return self._sol

    @sol.getter
    def sol(self):
        """returns answer to check"""
        return self._sol


class FirstMatrix(abstract):
    def __init__(self, random: bool = False, n: int = 3) -> None:
        super().__init__()
        self._mtr = None
        self._sol = np.random.randint(-10, 11, [n, 1]).astype("float")
        self._b = np.zeros([n, 1], dtype=np.double)

        if random:
            self._mtr = FirstMatrix._generator_random(n)

        else:
            self._mtr = np.array([[1 / (i + j + 1) for i in range(n)] for j in range(n)])
        self._find_b()

    def plu(self):
        """returns P, L, U components """
        n = self._mtr.shape[0]
        P, U, L = np.eye(n, dtype=np.double), self._mtr.copy(), np.eye(n, dtype=np.double)
        U.astype('float')
        for i in range(n):

            for k in range(i, n):
                if ~np.isclose(U[i, i], 0.0):
                    break
                U[[k, k + 1]] = U[[k + 1, k]]
                P[[k, k + 1]] = P[[k + 1, k]]

            factor = U[i + 1:, i] / U[i, i]
            L[i + 1:, i] = factor
            U[i + 1:] -= factor[:, np.newaxis] * U[i]

        return P, L, U

    @staticmethod
    def _forward_substitution(l1, b):

        n = l1.shape[0]

        y = np.zeros_like(b, dtype=np.double)

        y[0] = b[0] / l1[0, 0]

        for i in range(1, n):
            y[i] = (b[i] - np.dot(l1[i, :i], y[:i])) / l1[i, i]

        return y

    @staticmethod
    def _back_substitution(u, y):

        n = u.shape[0]

        x = np.zeros_like(y, dtype=np.double)

        x[-1] = y[-1] / u[-1, -1]

        for i in range(n - 2, -1, -1):
            x[i] = (y[i] - np.dot(u[i, i:], x[i:])) / u[i, i]

        return x

    def lu_solve(self):

        P, L, U = self.plu()

        y = self._forward_substitution(L, np.dot(P, self._b))

        return self._back_substitution(U, y)


class SecondMatrix(abstract):
    def __init__(self, random: bool = True, n: int = 3):
        super().__init__()
        self._mtr = None
        self._sol = np.random.randint(0, 15, [n, 1]).astype("float") + np.random.rand(n, 1)
        self._b = np.zeros([n, 1], dtype=np.double)

        if random:
            self._mtr = SecondMatrix._generator_random_spectral(n)

        else:
            self._mtr = np.array([[1 / (i * i + j * j + 1) for i in range(n)] for j in range(n)])
        self._find_b()

    @staticmethod
    def _generator_random_spectral(n) -> np.array:
        M = 50000000
        tmp1 = np.random.rand(n)
        maximum = np.max(tmp1)
        return np.array(
            [[tmp1[i] if i == j else np.random.randint(1, ((1 - maximum) * M / n - 1) // 1) / M for i in range(n)] for j
             in
             range(n)]).astype("float")

    def jacobi(self, tolerance=1e-10) -> np.ndarray:
        a = self._mtr
        b = self._b
        x = np.zeros_like(b)
        self.converge_check()
        for it_count in range(ITERATION_LIMIT):
            x_new = np.zeros_like(x)
            for i in range(a.shape[0]):
                s1 = np.dot(a[i, :i], x[:i])
                s2 = np.dot(a[i, i + 1:], x[i + 1:])
                x_new[i] = (b[i] - s1 - s2) / a[i, i]
            if np.allclose(x, x_new, atol=tolerance, rtol=0.):
                break
            x = x_new
            if it_count > ITERATION_LIMIT - 5:
                raise convergeError

        return x


class ThirdMatrix(abstract):
    def __init__(self, random: bool = True, n: int = 3):
        super().__init__()
        self._mtr = None
        self._sol = np.random.randint(0, 15, [n, 1]).astype("float") + np.random.rand(n, 1)
        self._b = np.zeros([n, 1], dtype=np.double)

        if random:
            self._mtr = ThirdMatrix._generator_random_spectral(n)

        else:
            self._mtr = np.array([[1 / (i * i + j * j + 1) for i in range(n)] for j in range(n)])
        self._find_b()

    @staticmethod
    def _generator_random_spectral(n) -> np.array:
        M = 50000000
        tmp1 = np.random.rand(n)
        maximum = np.max(tmp1)
        return np.array(
            [[tmp1[i] if i == j else np.random.randint(1, ((1 - maximum) * M / n - 1) // 1) / M for i in range(n)] for j
             in
             range(n)]).astype("float")

    def gauss_seidel(self, tolerance):

        A = self._mtr
        b = self._b
        m = A.shape[0]
        self.converge_check()
        x = np.zeros(m)
        x_n = np.zeros(m)

        iterations = 0

        while iterations < ITERATION_LIMIT:
            for i in range(0, m):
                x_n[i] = b[i] / A[i, i]
                tempsum = 0
                for j in range(0, m):
                    if j < i: tempsum += A[i, j] * x_n[j]
                    if j > i: tempsum += A[i, j] * x[j]
                x_n[i] -= tempsum / A[i, i]

            if np.linalg.norm(x - x_n, 2) < tolerance: break

            for i in range(0, m):
                x[i] = x_n[i]
            iterations += 1
        return x_n.reshape(m, 1)
