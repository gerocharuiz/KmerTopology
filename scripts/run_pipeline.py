"""
Crea los vecotres topologicos de una lista de archivos .fna
que tienen sus secuencias completas (solo tienen un contig).
Para esto ya se filtraron con el notebook explorar.ipynb y
los guarda

Con la matriz de vectores crea la matriz de distancias y 
la guarda


Autor: Gerardo Rocha Ruiz Jr
"""
# Para tener las rutas absolutas
from pathlib import Path
import sys

# carpeta raíz del proyecto
ROOT = Path.cwd().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from Bio import SeqIO
from pathlib import Path
from KmerTopology.matrix import create_matrix_vectors
from KmerTopology.distance import single_scale_distance

"""
Obtenemos los nombres de los archivos con secuencias
completas
"""
ROOT = Path(__file__).resolve().parent.parent

ruta = ROOT / "data" / "processed" / "secuencias_completas.csv"

df = pd.read_csv(ruta)

carpeta_data = Path("/files2/generic-amr/data/genomes") 

#Diccionario de Secuencias
secuences = {}

for nombre in df["Sec. Completas"]:
    ruta = carpeta_data / nombre
    secuence = list(SeqIO.parse(ruta, "fasta"))
    secuences[nombre] = (secuence[0].seq)

#Parametros
list_secuences = list(secuences.values())
kmers_size = 3
step_size = 4
max_step = 48


#Vectores topologicos de cada genoma
mat_B, mat_e = create_matrix_vectors(
    secuences = list_secuences, 
    kmers_size = kmers_size, 
    step_size = step_size, 
    max_step = max_step
)

#Guardamos
df_B = pd.DataFrame(mat_B)
df_e = pd.DataFrame(mat_e)
df_B.to_csv("../data/procssed/vectors_topB.csv")
df_e.to_csv("../data/procssed/vectors_tope.csv")
