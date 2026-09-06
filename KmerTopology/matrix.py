"""
create_matrix_vectors
Este algoritmo toma una lista de strings, que seran las secuencias de distintos genomas,
Una filtracion con las que se generarn los vectores topologicos de cada genoma.
Un int k para encontrar los k-meros.
Crea matriz de los vectores topologicos de cada uno

add_genomas_matrix
Si se desea agregrar un nuevo geneoma, se debe de dar la matriz ya creada al que se desea agregar
y la secuencia del genoma.

create_safe_matrix_vectors
Este algoritmo toma una lista de strings, que seran las secuencias de distintos genomas,
Una filtracion con las que se generarn los vectores topologicos de cada genoma.
Un int k para encontrar los k-meros.
Crea matriz de los vectores topologicos de cada uno
Ademas que las va guardando con cada paso que se calcule

Autor: Gerardo Rocha Ruiz Jr
"""

import numpy as np
import pandas as pd
from pathlib import Path
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
        #print(f"vect. topo de secuencia {i}/{n} calculado")
        #print(b)
        #print("***********************************************")
        #print(e)
        #print(f"vect. topo de secuencia {i} calculado y guardado")

    return mat_B, mat_lambda

#Lista de las secuencias, tamanio de los kmers, filtracion y archivo
def create_safe_matrix_vectors(
    secuences, 
    kmers_size, 
    step_size, 
    max_step, 
    ruta_B, 
    ruta_lambda):
    #Numero de genomas
    n = len(secuences)

    #Tamanio del vector topologico
    m = (max_step) * (4**kmers_size)

    #Creamos Carpetas
    ruta_B = Path(ruta_B)
    ruta_lambda = Path(ruta_lambda)

    # Creamos carpetas en caso de ser necesario
    ruta_B.parent.mkdir(parents=True, exist_ok=True)
    ruta_lambda.parent.mkdir(parents=True, exist_ok=True)
    
    # Crear archivos nuevos
    with open(ruta_B, "w") as f_B, open(ruta_lambda, "w") as f_lambda:

        for i, sequence in enumerate(secuences):
            print(f"Longitud de la secuencia {len(sequence)}")
            
            b, e = KmerTopology(
                sequence=sequence,
                kmers_size=kmers_size,
                step_size=step_size,
                max_step=max_step
            )

            # Escribir vector
            f_B.write(",".join(map(str, b)) + "\n")
            f_lambda.write(",".join(map(str, e)) + "\n")

            # Asegurar que se escriba físicamente
            f_B.flush()
            f_lambda.flush()

            print(f"[{i + 1:>3}/{n}] Vector topológico calculado")


    print("\nProceso terminado.")


"""
Agrega el vector topologico de una lista de genomas dada a una matriz 
de vectores topologicos ya hecha
"""
def add_genomas_matrix(mat, secuences, kmers_size, step_size, max_step):
    mat1 = create_matrix_vectors(secuences, kmers_size, step_size, max_step)
    return np.concatenate((mat, mat1), axis = 0)




