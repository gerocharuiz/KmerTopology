"""
create_matrix_vectors
Este algoritmo toma una lista de strings, que seran las secuencias de distintos genomas,
Una filtracion con las que se generarn los vectores topologicos de cada genoma.
Un int k para encontrar los k-meros.
Crea matriz de los vectores topologicos de cada uno

add_genomas_matrix
Si se desea agregrar un nuevo geneoma, se debe de dar la matriz ya creada al que se desea agregar
y la secuencia del genoma.


Autor: Gerardo Rocha Ruiz Jr
"""

import numpy as np
from KmerTopology.kmer_topology import KmerTopology, KmerHomology

#Lista de las secuencias, tamanio de los kmers, filtracion
def create_matrix_vectors(secuences, kmers_size, step_size, max_step):
    #Numero de genomas
    n = len(secuences)

    #Tamanio del vector topologico
    m = (max_step) * (4**kmers_size)

    #Matriz de nxm
    mat_B = np.zeros((n, m))
    mat_lambda = np.zeros((n, m))

    #Llenamos la matriz con los vectores topologicos
    for i in range(n):
        secuence = secuences[i]
        b, e = KmerTopology(
            sequence = secuence,
            kmers_size = kmers_size,
            step_size = step_size,
            max_step = max_step
        )
        mat_B[i] = b
        mat_lambda[i] = e

    return mat_B, mat_lambda


"""
Agrega el vector topologico de una lista de genomas dada a una matriz 
de vectores topologicos ya hecha
"""
def add_genomas_matrix(mat, secuences, kmers_size, step_size, max_step):
    mat1 = create_matrix_vectors(secuences, kmers_size, step_size, max_step)
    return np.concatenate((mat, mat1), axis = 0)




