"""
Dada la matriz de frecuencias de k-mers (salida de run_pipeline_freq.py)
crea la matriz de distancias y la guarda.

Es el equivalente de run_pipeline2.py para el baseline clasico.

Config de ejemplo:
{
    "mat_F": "data/processed/vectors_freq_escherichia.csv",
    "mat_DF": "data/processed/distance_F_escherichia.csv",
    "freq_metric": "single_scale"
}

"freq_metric" (opcional):
    "single_scale" (default) usa la misma single_scale_distance que
                   run_pipeline2.py, asi la comparacion cambia solo el
                   vector y no la distancia.
    O cualquier metrica de scipy.spatial.distance.pdist: "euclidean",
    "cosine", "braycurtis", "cityblock", "jensenshannon" (esta ultima solo
    tiene sentido con frecuencias normalizadas).

mat_F puede venir con o sin encabezado (run_pipeline_freq.py con "header"
true o false): si la primera fila son nombres de k-mer, se descarta sola.

Autor: Gerardo Rocha Ruiz Jr
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
from KmerTopology.distance import single_scale_distance


config_file = Path(sys.argv[1])
with open(config_file) as f:
    config = json.load(f)

"""
Direccion de la matriz de frecuencias de k-mers de los genomas
"""
ruta_matF = Path(config["mat_F"])
freq_metric = config.get("freq_metric", "single_scale")

df_F = pd.read_csv(ruta_matF, header=None)
# Si la primera fila son los nombres de los k-mers (header=true en
# run_pipeline_freq.py) la descartamos
num_F = df_F.apply(pd.to_numeric, errors="coerce")
if num_F.iloc[0].isna().all() and not num_F.iloc[1:].isna().any().any():
    num_F = num_F.iloc[1:]
mat_F = num_F.to_numpy(dtype=float)


# Ruta y nombre con los que se guardara la matriz de distancia
ruta_DF = Path(config["mat_DF"])


# Matriz de distancia de los vectores de frecuencias
if freq_metric == "single_scale":
    mat_DF = single_scale_distance(mat_F)
else:
    from scipy.spatial.distance import pdist, squareform
    mat_DF = squareform(pdist(mat_F, metric=freq_metric))

# Guardamos
df_DF = pd.DataFrame(mat_DF)
# Creamos carpetas en caso de ser necesario
ruta_DF.parent.mkdir(parents=True, exist_ok=True)

df_DF.to_csv(ruta_DF, index=False)
print(f"{ruta_matF.name} -> {ruta_DF} ({mat_DF.shape[0]}x{mat_DF.shape[1]}, metrica: {freq_metric})")
