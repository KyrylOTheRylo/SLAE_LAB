from classes import FirstMatrix


def lu_decomposition(t1: FirstMatrix = FirstMatrix()):
    print("Our matrix looks like")
    print(t1.get_matrix)
    print(f"our calculated solution:\n {t1.lu_solve()}")
    print(f"should be the answer\n{t1.sol}")


if __name__ == "__main__":
    c = FirstMatrix(True, 5)
    lu_decomposition(c)
