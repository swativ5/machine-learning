def matrix_multiplication(a, b):
    m1, n1 = len(a), len(a[0])
    m2, n2 = len(b), len(b[0])
    result = [[0] * n1 for _ in range(m1)]

    # matrix multiplication
    for row_a in range(m1):
        for col_b in range(n2):  
            sum = 0
            for col_a in range(n1):  
                sum += a[row_a][col_a] * b[col_a][col_b]
            result[row_a][col_b] = sum
    return result


# calls the matrix multiplication function m - 1 times to return the output
def A_m(A, m):
    result = A
    for _ in range(m - 1):
        result = matrix_multiplication(result, result)
    return result

A = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

print(A_m(A, 3))