"""
Crea la matriz de frecuencias de k-mers (baseline) de una lista de
genomas / contigs, para comparar contra los vectores topologicos de
run_pipeline.py.

Misma estructura de entrada que run_pipeline.py: dos modos, segun las
llaves presentes en el config JSON.

1) Modo original ("genomes_dir" + "csv_file"): toma el primer contig de
   cada .fna listado en csv_file (columna "name") dentro de genomes_dir.

2) Modo nuevo ("plasmid_manifest"): toma el CSV que produce
   03_build_plasmid_manifest.py y, por cada accession con
   final_fasta_path no vacio, se queda con el contig MAS LARGO de su
   fasta. Con "n_genomes" (+ "seed") limita cuantos genomas usar. Si
   usas la misma n_genomes y seed que en run_pipeline.py, se eligen
   exactamente los mismos genomas y en el mismo orden, asi que ambas
   matrices son directamente comparables fila a fila.

Config para el modo nuevo, ejemplo:
{
    "plasmid_manifest": "data/processed/selected_genomes/plasmid_manifest.csv",
    "n_genomes": 200,
    "seed": 42,
    "names_out": "data/processed/escherichia_plasmid_200.csv",
    "mat_F": "data/processed/vectors_freq_escherichia.csv",
    "kmer_size": 4,
    "normalize": true,
    "canonical": false,
    "header": false
}

Llaves propias del baseline:
  "kmer_size"  longitud k; el vector tiene dimension 4^k.
  "normalize"  (default true) frecuencias relativas; false = conteos crudos.
  "canonical"  (default false) junta cada k-mer con su reverso
               complementario; la dimension deja de ser 4^k.
  "header"     (default false) false = CSV sin encabezado (como los
               vectores topologicos que lee el notebook con header=None);
               true = las columnas se llaman como el k-mer (AAAA, AAAC,
               ...), que es el formato que lee pipeline_common.load_vectors
               (run_pipeline_knn.py / run_pipeline_pca.py).

"names_out" (opcional, solo modo nuevo) escribe un CSV con columna "name" =
accession, en el mismo orden que las filas de mat_F.

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
from KmerTopology.kmer_frequency import create_frequency_matrix, kmer_labels


config_file = Path(sys.argv[1])
with open(config_file) as f:
    config = json.load(f)

"""
Parametros
"""
kmers_size = config["kmer_size"]
normalize = config.get("normalize", True)
canonical = config.get("canonical", False)
header = config.get("header", False)

# Ruta con la que se guardara la matriz
ruta_matF = Path(config["mat_F"])


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

# Vector de frecuencias de k-mers de cada genoma
mat_F = create_frequency_matrix(
    secuences = list_secuences,
    kmers_size = kmers_size,
    normalize = normalize,
    canonical = canonical
)
print(f"Matriz de frecuencias: {mat_F.shape[0]} secuencias x {mat_F.shape[1]} k-mers (k={kmers_size})")

# Guardamos
df_F = pd.DataFrame(mat_F, columns=kmer_labels(kmers_size, canonical) if header else None)
# Creamos carpetas en caso de ser necesario
ruta_matF.parent.mkdir(parents=True, exist_ok=True)
df_F.to_csv(ruta_matF, index=False, header=header)
