"""
Codigo para generar un arbol filogenetico a partir de la matriz de
distancias del baseline de frecuencias de k-mers (salida de
run_pipeline_freq2.py).

Es el equivalente de run_pipeline3.py para el baseline clasico.

Se necesita un archivo JSON que contenga:
- Direccion de la matriz de distancias de frecuencias.
- Direccion del archivo CSV con los nombres.
- Direccion donde se guardara el arbol.

Config de ejemplo:
{
    "mat_DF": "data/processed/distance_F_escherichia.csv",
    "nombres_csv": "data/processed/escherichia_plasmid_200.csv",
    "TreeF": "results/figures/pruebas/tree_F_escherichia.png"
}

Autor: Gerardo Rocha Ruiz Jr.
"""

# Para tener las rutas absolutas
from pathlib import Path
import sys

# carpeta raíz del proyecto (se prueba tanto la carpeta actual como su padre,
# segun desde donde se corra el script)
for ROOT in (Path.cwd(), Path.cwd().parent):
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
Direccion de la matriz de distancia y de los nombres de los archivos
"""
ruta_DF = Path(config["mat_DF"])
ruta_names = Path(config["nombres_csv"])

df_F = pd.read_csv(ruta_DF)
df_names = pd.read_csv(ruta_names)

labels = []
for nombre in df_names["name"]:
    if nombre.endswith(".fna"):
        labels.append(nombre[:-4])
    else:
        labels.append(nombre)

if df_F.shape[0] != len(labels):
    sys.exit(f"La matriz de distancias tiene {df_F.shape[0]} filas pero "
             f"'nombres_csv' tiene {len(labels)} nombres")


# Ruta y nombre con los que se guardara el arbol filogenetico
ruta_TreeF = Path(config["TreeF"])


#Obtenemos la figura
mat_DF = df_F.to_numpy()
figF, axesF = UPGMA(mat_DF, labels, ruta_TreeF)

# Guardamos
ruta_TreeF.parent.mkdir(parents=True, exist_ok=True)

figF.savefig(ruta_TreeF.with_suffix(".png"), dpi=300)
print(f"Arbol guardado en {ruta_TreeF.with_suffix('.png')}")
