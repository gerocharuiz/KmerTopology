"""
Crea los vectores topologicos de una lista de genomas / contigs.

Tiene dos modos, segun las llaves presentes en el config JSON:

1) Modo original ("genomes_dir" + "csv_file"): toma el primer contig de
   cada .fna listado en csv_file (columna "name") dentro de genomes_dir.
   Este modo no cambio, para no romper configs viejos (p.ej.
   vect_escherichia_10.json).

2) Modo nuevo ("plasmid_manifest"): toma como entrada el CSV que produce
   03_build_plasmid_manifest.py. Para cada accession con
   final_fasta_path no vacio, abre ese fasta (que puede tener 1 o varios
   contigs plasmidicos, segun si 03 se corrio con "largest_only") y se
   queda con el contig MAS LARGO de ese archivo -- ya no con el "primer
   contig" arbitrario. Opcionalmente puedes limitar cuantos genomas usar
   con "n_genomes" (y fijar "seed" para que la muestra sea reproducible;
   si no das seed, se usa n_genomes en el orden en que vienen en el
   manifiesto).

Config para el modo nuevo, ejemplo:
{
    "plasmid_manifest": "plasmid_manifest.csv",
    "n_genomes": 200,
    "seed": 42,
    "names_out": "data/processed/escherichia_plasmid_200.csv",
    "mat_B": "data/processed/vectors_topB_escherichia.csv",
    "mat_E": "data/processed/vectors_topE_escherichia.csv",
    "kmer_size": 4,
    "step_size": 4,
    "max_step": 48
}

"names_out" (opcional) escribe un CSV con columna "name" = accession, en
el mismo formato que espera run_pipeline3.py para las etiquetas del arbol.

Autor: Gerardo Rocha Ruiz Jr
"""
# Para tener las rutas absolutas
from pathlib import Path
import sys

# carpeta raíz del proyecto
ROOT = Path.cwd()

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
Parametros
"""
kmers_size = config["kmer_size"]
step_size = config["step_size"]
max_step = config["max_step"]

# Ruta y nombres con los que se guadaran las matrices
ruta_matB = Path(config["mat_B"])
ruta_matE = Path(config["mat_E"])


def contig_mas_largo(fasta_path):
    """Regresa el Seq del contig mas largo dentro de un fasta (1 o varios registros)."""
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if not records:
        return None
    return max(records, key=lambda r: len(r.seq)).seq


secuences = {}

if "plasmid_manifest" in config:
    """
    Modo nuevo: entrada = plasmid_manifest.csv de 03_build_plasmid_manifest.py
    """
    df_manifest = pd.read_csv(config["plasmid_manifest"])
    # Solo genomas donde Platon SI encontro contig(s) plasmidico(s)
    df_manifest = df_manifest[df_manifest["final_fasta_path"].notna()]
    df_manifest = df_manifest[df_manifest["final_fasta_path"] != ""]

    n_genomes = config.get("n_genomes")
    seed = config.get("seed")
    if n_genomes is not None and n_genomes < len(df_manifest):
        if seed is not None:
            df_manifest = df_manifest.sample(n=n_genomes, random_state=seed)
        else:
            df_manifest = df_manifest.head(n_genomes)

    n_sin_contigs = 0
    for _, row in df_manifest.iterrows():
        accession = row["accession"]
        fasta_path = Path(row["final_fasta_path"])
        seq = contig_mas_largo(fasta_path)
        if seq is None:
            n_sin_contigs += 1
            continue
        secuences[accession] = seq

    print(f"Genomas usados: {len(secuences)}")
    if n_sin_contigs:
        print(f"Genomas con fasta vacio/ilegible (omitidos): {n_sin_contigs}")

    names_out = config.get("names_out")
    if names_out:
        ruta_names = Path(names_out)
        ruta_names.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"name": list(secuences.keys())}).to_csv(ruta_names, index=False)

else:
    """
    Modo original: entrada = genomes_dir + csv_file (columna "name"),
    se usa solo el primer contig de cada .fna.
    """
    carpeta_data = Path(config["genomes_dir"])
    archivo_csv = Path(config["csv_file"])
    df = pd.read_csv(archivo_csv)

    for nombre in df["name"]:
        ruta = carpeta_data / nombre
        secuence = list(SeqIO.parse(ruta, "fasta"))
        secuences[nombre] = secuence[0].seq


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
