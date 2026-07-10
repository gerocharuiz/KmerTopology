"""
Dada una matriz de vectores crea la matriz de distancias y 
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
import json
from KmerTopology.distance import single_scale_distance


config_file = Path(sys.argv[1])
with open(config_file) as f:
    config = json.load(f)

"""
Direccion de las matrices que contienen a los vectores topologicos
de los genomas
"""
ruta_matB = Path(config["mat_B"])
ruta_matE = Path(config["mat_E"])
df_B = pd.read_csv(ruta_matB)
df_E = pd.read_csv(ruta_matE)


# Ruta y nombres con los que se guadaran las matrices de distancia
ruta_DB = Path(config["mat_DB"])
ruta_DE = Path(config["mat_DE"])


# Matrices de distancia de los vectores topologicos
mat_DB = single_scale_distance(df_B.to_numpy())
mat_DE = single_scale_distance(df_E.to_numpy())

# Guardamos
df_DB = pd.DataFrame(mat_DB)
df_DE = pd.DataFrame(mat_DE)
# Creamos carpetas en caso de ser necesario
ruta_DB.parent.mkdir(parents=True, exist_ok=True)
ruta_DE.parent.mkdir(parents=True, exist_ok=True)

df_DB.to_csv(ruta_DB, index=False)
df_DE.to_csv(ruta_DE, index=False)