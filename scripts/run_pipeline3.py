"""
Código para generar un árbol filogenético a partir de una
matriz de distancias.

Se necesita un archivo JSON que contenga:
- Dirección de la matriz de distancias Betti.
- Dirección de la matriz de distancias Eigenvalues.
- Dirección del archivo CSV con los nombres.
- Dirección donde se guardarán los árboles.

Autor: Gerardo Rocha Ruiz Jr.
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
from KmerTopology.tree import UPGMA


config_file = Path(sys.argv[1])
with open(config_file) as f:
    config = json.load(f)

"""
Direccion de las matrices que contienen a los vectores topologicos
de los genomas
Direccion de la matriz de distancia y de los nombres de los archivos
"""
ruta_DB = Path(config["mat_DB"])
ruta_DE = Path(config["mat_DE"])
ruta_names = Path(config["nombres_csv"])

df_B = pd.read_csv(ruta_DB)
df_E = pd.read_csv(ruta_DE)
df_names = pd.read_csv(ruta_names)

labels = []
for nombre in df_names["name"]:
    if nombre.endswith(".fna"):
        labels.append(nombre[:-4])
    else:
        labels.append(nombre)


# Ruta y nombres con los que se guadaran los arboles filogenticos
ruta_TreeB = Path(config["TreeB"])
ruta_TreeE = Path(config["TreeE"])


#Obtenemos las figuras
mat_DB = df_B.to_numpy()
mat_DE = df_E.to_numpy()
figB, axesB = UPGMA(mat_DB, labels, ruta_TreeB)
figE, axesE = UPGMA(mat_DE, labels, ruta_TreeE)

# Guardamos
ruta_TreeB.parent.mkdir(parents=True, exist_ok=True)
ruta_TreeE.parent.mkdir(parents=True, exist_ok=True)

figB.savefig(ruta_TreeB.with_suffix(".png"), dpi=300)
figE.savefig(ruta_TreeE.with_suffix(".png"), dpi=300)