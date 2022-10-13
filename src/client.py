from classes import FirstMatrix, SecondMatrix, ThirdMatrix
from customexceptions import convergeError


def lu_decomposition(t1: FirstMatrix = FirstMatrix()):
    print("Our matrix looks like")
    print(t1.get_matrix)
    print(f"our calculated solution:\n {t1.lu_solve()}")
    print(f"should be the answer\n{t1.sol}")
    print("END--------------------------------------------FIRST\n\n")


def jacobi_method(t1: SecondMatrix = SecondMatrix(), tolerance: float = 0.01):
    print(f"Our Matrix is:\n {t1.get_matrix}\n")
    print(f"Expected answer: \n {t1.sol}")
    tmp = t1.jacobi(tolerance)
    print(f"our answer:\n {tmp}")
    print(f"delta is: \n{t1.sol - tmp}")
    print("END--------------------------------------------SECOND\n\n")


def seidel(t1: ThirdMatrix = ThirdMatrix(), tolerance: float = 0.0001):
    print(f"Our Matrix is:\n {t1.get_matrix}\n")
    print(f"Expected answer: \n {t1.sol}")
    tmp = t1.gauss_seidel(tolerance)
    print(f"our answer:\n {tmp}")
    print(f"delta is: \n{t1.sol - tmp}")
    print("END--------------------------------------------THIRD\n\n")


if __name__ == "__main__":
    c = FirstMatrix(True, 12)
    lu_decomposition(c)
    try:
        c1 = SecondMatrix(True, 6)
        jacobi_method(c1, 1e-11)
    except convergeError:
        print("======================================\nConverge error In Jacobi ")
    try:
        c2 = ThirdMatrix(True, 4)
        seidel(c2, 1e-5)
    except convergeError:
        print("======================================\nConverge error In SEIDEL ")
