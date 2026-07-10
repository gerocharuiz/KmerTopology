"""
Crea los vectores topologicos de una lista de archivos .fna
dados, sólo saca los vectores topologicos del primer contig
del archivo .fna


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
import json
from Bio import SeqIO
from KmerTopology.matrix import create_matrix_vectors
from KmerTopology.distance import single_scale_distance


config_file = Path(sys.argv[1])
with open(config_file) as f:
    config = json.load(f)

"""
Dirección de la carpeta de la base de datos

"""
carpeta_data = Path(config["genomes_dir"])
# "/files2/generic-amr/data/genomes"


"""
Direccion del archivo que contiene los nombres de los datos con los
que se quiere trabajar de la carpeta de la base de datos dada
"""
archivo_csv = Path(config["csv_file"])
df = pd.read_csv(archivo_csv)
#ruta = ROOT / "data" / "processed" / archivo


# Ruta y nombres con los que se guadaran las matrices
ruta_matB = Path(config["mat_B"])
ruta_matE = Path(config["mat_E"])

# Parametros
kmers_size = config["kmer_size"]
step_size = config["step_size"]
max_step = config["max_step"]
 

#Diccionario de Secuencias
secuences = {}

for nombre in df["Sec. Completas"]:
    ruta = carpeta_data / nombre
    secuence = list(SeqIO.parse(ruta, "fasta"))
    secuences[nombre] = (secuence[0].seq)


list_secuences = list(secuences.values())



#Vectores topologicos de cada genoma
mat_B, mat_e = create_matrix_vectors(
    secuences = list_secuences, 
    kmers_size = kmers_size, 
    step_size = step_size, 
    max_step = max_step
)

# Guardamos
df_B = pd.DataFrame(mat_B)
df_e = pd.DataFrame(mat_e)
# Creamos carpetas en caso de ser necesario
ruta_matB.parent.mkdir(parents=True, exist_ok=True)
ruta_matE.parent.mkdir(parents=True, exist_ok=True)

df_B.to_csv(ruta_matB, index=False)
df_e.to_csv(ruta_matE, index=False)