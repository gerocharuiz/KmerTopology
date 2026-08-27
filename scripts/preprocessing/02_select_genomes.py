#!/usr/bin/env python3
"""
02_run_platon_batch.py

Corre Platon sobre cada .fna listado en el manifiesto generado por
01_select_genomes.py, y deja los resultados en <outdir>/<accession>/.

Requiere Platon instalado y su base de datos descargada:
    conda install -c bioconda -c conda-forge platon
    platon --db <ruta_db_descargada> ...   (o exporta PLATON_DB env var)

Salida por genoma (nombres que asigna Platon, basados en el nombre del
archivo de entrada -- por eso el .fna debe llamarse <accession>.fna):
    <accession>/<accession>.chromosome.fasta
    <accession>/<accession>.plasmid.fasta   <-- esto es lo que nos interesa
    <accession>/<accession>.tsv
    <accession>/<accession>.json

Uso:
    python 02_run_platon_batch.py \
        --manifest selected_genomes.csv \
        --platon-db /ruta/a/platon-db \
        --outdir platon_out \
        --threads 8
"""
import argparse
import csv
import os
import subprocess
import sys


def already_done(accession: str, outdir: str) -> bool:
    tsv = os.path.join(outdir, accession, f"{accession}.tsv")
    return os.path.isfile(tsv)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True,
                     help="CSV de salida de 01_select_genomes.py")
    ap.add_argument("--platon-db", required=True,
                     help="Ruta a la base de datos de Platon ya descargada")
    ap.add_argument("--outdir", default="platon_out")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None,
                help="Máximo de genomas a procesar")
    ap.add_argument("--platon-bin", default="platon",
                     help="Nombre/ruta del ejecutable de platon")
    ap.add_argument("--mode", default="sensitivity",
                     choices=["sensitivity", "accuracy", "specificity"],
                     help="Modo de clasificacion de Platon (ver su documentacion)")
    ap.add_argument("--log", default="platon_run_log.csv")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.manifest, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["found"] in ("True", "true", "1")]
    if args.limit is not None:
        rows = rows[:args.limit]

    print(f"Genomas a procesar: {len(rows)}")

    log_rows = []
    for i, row in enumerate(rows, 1):
        accession = row["accession"]
        fna_path = row["fna_path"]
        genome_outdir = os.path.join(args.outdir, accession)

        if already_done(accession, args.outdir):
            print(f"[{i}/{len(rows)}] {accession}: ya procesado, se omite")
            log_rows.append({"accession": accession, "status": "skipped_already_done", "stderr_tail": ""})
            continue

        os.makedirs(genome_outdir, exist_ok=True)
        cmd = [
            args.platon_bin,
            "--db", args.platon_db,
            "--output", genome_outdir,
            "--threads", str(args.threads),
            "--mode", args.mode,
            fna_path,
        ]
        print(f"[{i}/{len(rows)}] {accession}: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode == 0:
                status = "ok"
            else:
                status = f"error_returncode_{result.returncode}"
            stderr_tail = result.stderr[-500:] if result.stderr else ""
        except subprocess.TimeoutExpired:
            status = "timeout"
            stderr_tail = ""
        except FileNotFoundError:
            print(f"No se encontro el ejecutable de Platon ('{args.platon_bin}'). "
                  f"Verifica instalacion/activacion del entorno conda.", file=sys.stderr)
            sys.exit(1)

        log_rows.append({"accession": accession, "status": status, "stderr_tail": stderr_tail})

    with open(args.log, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["accession", "status", "stderr_tail"])
        writer.writeheader()
        writer.writerows(log_rows)

    n_ok = sum(1 for r in log_rows if r["status"] == "ok")
    n_skip = sum(1 for r in log_rows if r["status"] == "skipped_already_done")
    n_fail = len(log_rows) - n_ok - n_skip
    print(f"\nCompletados OK: {n_ok} | Omitidos (ya hechos): {n_skip} | Fallidos: {n_fail}")
    print(f"Log detallado en: {args.log}")


if __name__ == "__main__":
    main()
