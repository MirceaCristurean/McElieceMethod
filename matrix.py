from random import shuffle
import numpy as np
from time import time

timeout = time() + 3  # seconds


def random_vector(length, weight):
    arr = [True] * weight + [False] * (length - weight)
    shuffle(arr)
    return np.array(arr, dtype='bool')


def generate_matrix_h(rows, columns, weight):
    # initialize matrix with 0 binary
    matrix = np.zeros((rows, columns), dtype='bool')
    for index in range(0, rows):
        new_row = [True] * weight + [False] * (columns - weight)
        shuffle(new_row)
        matrix[index] = list(new_row)
    return matrix


def get_rank(m):
    return np.linalg.matrix_rank(m)


def print_matrix(matrix, binary=True):
    if binary:
        matrix = matrix * 1
        for row in matrix:
            print(row)
    else:
        for row in matrix:
            print(row)
    print()


def print_rank(matrix):
    print("The rank is ", np.linalg.matrix_rank(matrix))


def remove_duplicates_rows(matrix):
    h_final = np.copy(matrix)
    delta = n - k - get_rank(matrix)
    if delta != 0:
        h_temp = np.copy(matrix)
        while delta > 0:
            for column in range(0, len(h_temp)):
                h_temp = np.delete(h_temp, column, 0)
                if get_rank(matrix) == get_rank(h_temp):
                    matrix = np.copy(h_temp)
                    h_final = np.copy(h_temp)
                break
            else:
                h_temp = np.copy(matrix)
        delta = delta - 1
    else:
        print('Rank full')
    print("Matrix without duplicate rows")
    print_matrix(h_final)
    return h_final


def calculate_score(matrix, syndrome):
    ls, = syndrome.nonzero()
    l_score = []
    for index in range(0, n):
        nonzero, = matrix[:, index].nonzero()
        intersect = np.intersect1d(nonzero, ls)
        l_score.append(len(intersect))
    print("score", l_score)
    return l_score


def decode(matrix, syndrome, e):
    y = np.zeros(n, dtype="bool")
    length, = syndrome.nonzero()
    print("length", length)
    while len(length) != 0:
        score = calculate_score(matrix, syndrome)
        m = max(score)
        list_max, = (np.array(score) >= m).nonzero()
        v = np.zeros(n - k, dtype="bool")
        x = np.zeros(n, dtype="bool")
        for column in list_max:
            v = v + np.transpose(matrix)[:][column]
            x[column] = 1
        syndrome = np.logical_xor(syndrome, v)
        y = y + x
        length, = syndrome.nonzero()
        if time() > timeout:
            return False
    return np.array_equal(e, y)


if __name__ == '__main__':
    validity = []
    n = 20  # columns
    k = 10  # n-k rows
    w = 5  # number of 1 value on a row
    errorWeight = 3  # complexity of error vector
    generated_matrix = generate_matrix_h(n - k, n, w)  # generating random matrix
    striped_matrix = remove_duplicates_rows(generated_matrix)  # cleaning matrix => generator matrix H
    start_error_weight = 3
    stop_error_weight = 4
    number_of_retries = 2
    for errorWeight in range(start_error_weight, stop_error_weight):
        true_count = 0
        for i in range(1, number_of_retries):
            print(f"Weight: {errorWeight}, i: {i}")
            error_vector = random_vector(n, errorWeight)  # error vector
            S = striped_matrix @ error_vector  # calculating syndrome
            calculated_syndrome = np.copy(S)
            print(decode(striped_matrix, calculated_syndrome, error_vector))
            if (decode(striped_matrix, calculated_syndrome, error_vector)) is True:
                true_count += 1
        result = true_count / number_of_retries
        validity.append(result)
    print(validity)
