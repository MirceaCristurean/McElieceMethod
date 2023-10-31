from random import shuffle, randrange
import numpy as np
from time import time
import sys
timeout = time() + 3  # seconds

def random_vector(length, w):
    arr = [True]*w+[False]*(length-w)
    shuffle(arr)
    return np.array(arr, dtype='bool')

def generate_matrix_h(rows, columns, w):
    # initializare matrice cu zero binar
    matrix = np.zeros((rows, columns), dtype='bool')
    for index in range(0, rows):
        new_row = [True]*w+[False]*(columns-w)
        shuffle(new_row)
        matrix[index] = list(new_row)
    print_matrix(matrix)
    return matrix

def get_rank(m):
    return np.linalg.matrix_rank(m)

def print_matrix(matrix, binary=True):
    if binary:
        matrix = matrix*1
        for row in matrix:
            print(row)
    else:
        for row in matrix:
            print(row)
    print()

def print_rank(matrix):
    print("The rank is ", np.linalg.matrix_rank(matrix))

def remove_duplicates_rows(matrix):
    H_final = np.copy(matrix)
    print_matrix(matrix)
    delta = n-k-get_rank(matrix)
    if delta != 0:
        H_temp = np.copy(matrix)
        while (delta > 0):
            for column in range(0, len(H_temp)):
                H_temp = np.delete(H_temp, column, 0)
                print_matrix(H_temp)
                if (get_rank(matrix) == get_rank(H_temp)):
                    matrix = np.copy(H_temp)
                    H_final = np.copy(H_temp)
                break
            else:
                H_temp = np.copy(matrix)
        delta = delta - 1
    else:
        print('Rank full')
    print_matrix(H_final)
    return H_final

def calcul_Lscor(matrix, sindrome):
    LS, = sindrome.nonzero()
    Lscor = []
    for index in range(0, n):
        nonzeros, = matrix[:, index].nonzero()
        intersect = np.intersect1d(nonzeros, LS)
        Lscor.append(len(intersect))
    print("Scor", Lscor)
    return Lscor

def decodaj(matrix, sindrome, e):
    y = np.zeros(n, dtype="bool")
    length, = sindrome.nonzero()
    while len(length) != 0:
        # print(length)
        scor = calcul_Lscor(matrix, sindrome)
        M = max(scor)
        ListMax, = (np.array(scor) >= M).nonzero()
        V = np.zeros(n-k, dtype="bool")
        x = np.zeros(n, dtype="bool")
        for column in ListMax:
            V = V + np.transpose(matrix)[:][column]
            x[column] = 1
        sindrome = np.logical_xor(sindrome, V)
        y = y + x
        length, = sindrome.nonzero()
        if time() > timeout:
            return False
    return np.array_equal(e,y)


if __name__ == '__main__':
    validity = []
    n = 200  # coloane
    k = 100   # n-k randuri
    w = 5  # numar de 1 pe linie
    errorWeight = 3 # complexitatea vectorului de eroare
    generated_matrix = generate_matrix_h(n-k, n, w) # generare aleatorie matrice
    striped_matrix = remove_duplicates_rows(generated_matrix) # curatare matrice => matricea generatoare H
    for errorWeight in range(3, 4):
        true_contor = 0
        for i in range(1,2):
            print(f"Pondere: {errorWeight}, i: {i}")
            e = random_vector(n, errorWeight) # vectorul de eroare de pondere errorWeight
            S = striped_matrix @ e # calcul sindrom
            sindrome = np.copy(S)
            print(decodaj(striped_matrix, sindrome, e))
            if (decodaj(striped_matrix, sindrome, e)) is True:
                true_contor += 1
        result = true_contor/5
        validity.append(result)
    print(validity)

