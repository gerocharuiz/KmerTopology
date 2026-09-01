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

Uso:
    python 01_select_genomes.py \
        --metadata pangenome_metadata.csv \
        --fna-dir /ruta/a/mis/fnas \
        --species "Escherichia coli" \
        --antibiotic ciprofloxacin \
        --out selected_genomes.csv
"""
import argparse
import csv
import os
import sys


def find_fna(accession: str, fna_dir: str) -> str | None:
    """Busca <accession>.fna o <accession>.fna.gz dentro de fna_dir."""
    for ext in (".fna", ".fna.gz", ".fasta", ".fasta.gz"):
        candidate = os.path.join(fna_dir, f"{accession}{ext}")
        if os.path.isfile(candidate):
            return candidate
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata", required=True, help="pangenome_metadata.csv")
    ap.add_argument("--fna-dir", required=True,
                     help="Directorio donde ya tienes descargados los .fna")
    ap.add_argument("--species", default=None,
                     help="Filtra por columna 'species' (match exacto). Ej: 'Escherichia coli'")
    ap.add_argument("--antibiotic", default=None,
                     help="Filtra por columna 'antibiotic' (match exacto). Ej: ciprofloxacin")
    ap.add_argument("--dataset", default=None,
                     help="Filtra por columna 'dataset' si quieres solo train/test de un corte")
    ap.add_argument("--out", default="selected_genomes.csv")
    args = ap.parse_args()

    n_total = 0
    n_kept = 0
    n_found = 0
    rows_out = []

    with open(args.metadata, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_total += 1
            if args.species and row["species"] != args.species:
                continue
            if args.antibiotic and row["antibiotic"] != args.antibiotic:
                continue
            if args.dataset and args.dataset not in row["dataset"].split():
                continue
            n_kept += 1

            accession = row["accession"]
            fna_path = find_fna(accession, args.fna_dir)
            if fna_path:
                n_found += 1

            rows_out.append({
                "accession": accession,
                "species": row["species"],
                "antibiotic": row["antibiotic"],
                "phenotype": row["phenotype"],
                "adjusted_phenotype": row.get("adjusted_phenotype", ""),
                "fna_path": fna_path or "",
                "found": bool(fna_path),
            })

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["accession", "species", "antibiotic", "phenotype",
                        "adjusted_phenotype", "fna_path", "found"],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Filas totales en metadata:      {n_total}")
    print(f"Filas que pasan el filtro:      {n_kept}")
    print(f"  .fna localizados:             {n_found}")
    print(f"  .fna faltantes:               {n_kept - n_found}")
    print(f"Manifiesto escrito en:          {args.out}")

    if n_kept - n_found > 0:
        print("\nATENCION: hay accessions sin .fna encontrado en --fna-dir.",
              file=sys.stderr)
        print("Revisa que el nombre de archivo sea exactamente <accession>.fna",
              file=sys.stderr)


if __name__ == "__main__":
    main()
