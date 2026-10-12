    #!/usr/bin/env python3
"""
01_select_genomes.py

Filtra pangenome_metadata.csv por especie/antibiotico (opcional) y localiza
el archivo .fna correspondiente a cada accession dentro de un directorio local.

IMPORTANTE: Platon nombra sus archivos de salida a partir del nombre del
archivo de entrada (sin extension), NO a partir del accession en sí. Por eso
este script exige (o renombra simbólicamente) que cada .fna se llame
"<accession>.fna" -- así el nombre de salida de Platon coincide con el
accession y el script 03 puede volver a asociarlos sin ambigüedad.

Uso (config JSON, mismo estilo que run_pipeline.py / run_pipeline2.py / run_pipeline3.py):
    python 01_select_genomes.py config_select.json

Donde config_select.json luce asi (todas las llaves menos "metadata" y
"fna_dir" son opcionales):
{
    "metadata": "pangenome_metadata.csv",
    "fna_dir": "/ruta/a/mis/fnas",
    "species": "Escherichia coli",
    "antibiotic": "ciprofloxacin",
    "dataset": null,
    "mic_column": "mic",
    "out": "selected_genomes.csv"
}

"mic_column" (opcional, default "mic") es el nombre de la columna de MIC
(concentracion minima inhibitoria) dentro de pangenome_metadata.csv. Se
copia tal cual al manifiesto de salida junto con phenotype, para poder
usarla mas adelante como variable a predecir en vez de la etiqueta
categorica resistente/susceptible/intermedio.
"""
import csv
import json
import os
import sys
from pathlib import Path


def find_fna(accession: str, fna_dir: str) -> str | None:
    """Busca <accession>.fna o <accession>.fna.gz dentro de fna_dir."""
    for ext in (".fna", ".fna.gz", ".fasta", ".fasta.gz"):
        candidate = os.path.join(fna_dir, f"{accession}{ext}")
        if os.path.isfile(candidate):
            return candidate
    return None


def main():
    if len(sys.argv) != 2:
        print("Uso: python 01_select_genomes.py config.json", file=sys.stderr)
        sys.exit(1)

    config_file = Path(sys.argv[1])
    with open(config_file, encoding="utf-8") as f:
        config = json.load(f)

    metadata = config["metadata"]
    fna_dir = config["fna_dir"]
    species = config.get("species")
    antibiotic = config.get("antibiotic")
    dataset = config.get("dataset")
    mic_column = config.get("mic_column", "mic")
    out = config.get("out", "selected_genomes.csv")

    n_total = 0
    n_kept = 0
    n_found = 0
    n_mic = 0
    rows_out = []

    with open(metadata, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_total += 1
            if species and row["species"] != species:
                continue
            if antibiotic and row["antibiotic"] != antibiotic:
                continue
            if dataset and dataset not in row["dataset"].split():
                continue
            n_kept += 1

            accession = row["accession"]
            fna_path = find_fna(accession, fna_dir)
            if fna_path:
                n_found += 1

            mic = row.get(mic_column, "")
            if mic not in ("", None):
                n_mic += 1

            rows_out.append({
                "accession": accession,
                "species": row["species"],
                "antibiotic": row["antibiotic"],
                "phenotype": row["phenotype"],
                "adjusted_phenotype": row.get("adjusted_phenotype", ""),
                "mic": mic,
                "fna_path": fna_path or "",
                "found": bool(fna_path),
            })

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["accession", "species", "antibiotic", "phenotype",
                        "adjusted_phenotype", "mic", "fna_path", "found"],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Filas totales en metadata:      {n_total}")
    print(f"Filas que pasan el filtro:      {n_kept}")
    print(f"  .fna localizados:             {n_found}")
    print(f"  .fna faltantes:               {n_kept - n_found}")
    print(f"  MIC ('{mic_column}') presente: {n_mic}")
    print(f"  MIC faltante:                 {n_kept - n_mic}")
    print(f"Manifiesto escrito en:          {out}")

    if n_kept - n_found > 0:
        print("\nATENCION: hay accessions sin .fna encontrado en fna_dir.",
              file=sys.stderr)
        print("Revisa que el nombre de archivo sea exactamente <accession>.fna",
              file=sys.stderr)

    if n_kept - n_mic > 0:
        print(f"\nATENCION: hay accessions sin MIC en la columna '{mic_column}' "
              f"de pangenome_metadata.csv. Si el nombre de tu columna de MIC es "
              f"otro, ajusta 'mic_column' en el config.", file=sys.stderr)


if __name__ == "__main__":
    main()
