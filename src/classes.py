import numpy as np
from abc import ABC


class abstract(ABC):
    def __init__(self):
        """initialization for a class"""
        self._mtr = None
        self._sol = None

    @staticmethod
    def _generator_random(n) -> np.array:
        return np.random.randint(-20, 21, [n, n]).astype("float")

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
        self._b = np.zeros([4, 1], dtype=np.double)

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

        # Number of rows
        n = u.shape[0]

        # Allocating space for the solution vector
        x = np.zeros_like(y, dtype=np.double)

        # Here we perform the back-substitution.
        # Initializing with the last row.
        x[-1] = y[-1] / u[-1, -1]

        # Looping over rows in reverse (from the bottom up),
        # starting with the second to last row, because the
        # last row solve was completed in the last step.
        for i in range(n - 2, -1, -1):
            x[i] = (y[i] - np.dot(u[i, i:], x[i:])) / u[i, i]

        return x

    def lu_solve(self):

        P, L, U = self.plu()

        y = self._forward_substitution(L, np.dot(P, self._b))

        return self._back_substitution(U, y)
