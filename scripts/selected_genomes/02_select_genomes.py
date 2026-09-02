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

Uso (config JSON, mismo estilo que run_pipeline.py / run_pipeline2.py / run_pipeline3.py):
    python 02_run_platon_batch.py config_platon.json

Donde config_platon.json luce asi (solo "manifest" y "platon_db" son
obligatorias, el resto tiene los mismos valores por default que la version
con argparse):
{
    "manifest": "selected_genomes.csv",
    "platon_db": "/ruta/a/platon-db",
    "outdir": "platon_out",
    "threads": 8,
    "limit": null,
    "platon_bin": "platon",
    "mode": "sensitivity",
    "log": "platon_run_log.csv"
}
"""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def already_done(accession: str, outdir: str) -> bool:
    tsv = os.path.join(outdir, accession, f"{accession}.tsv")
    return os.path.isfile(tsv)


def main():
    if len(sys.argv) != 2:
        print("Uso: python 02_run_platon_batch.py config.json", file=sys.stderr)
        sys.exit(1)

    config_file = Path(sys.argv[1])
    with open(config_file, encoding="utf-8") as f:
        config = json.load(f)

    manifest = config["manifest"]
    platon_db = config["platon_db"]
    outdir = config.get("outdir", "platon_out")
    threads = config.get("threads", 4)
    limit = config.get("limit")
    platon_bin = config.get("platon_bin", "platon")
    mode = config.get("mode", "sensitivity")
    if mode not in ("sensitivity", "accuracy", "specificity"):
        print(f"'mode' invalido en config: {mode!r} "
              f"(valores validos: sensitivity, accuracy, specificity)", file=sys.stderr)
        sys.exit(1)
    log = config.get("log", "platon_run_log.csv")

    os.makedirs(outdir, exist_ok=True)

    with open(manifest, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["found"] in ("True", "true", "1")]
    if limit is not None:
        rows = rows[:limit]

    print(f"Genomas a procesar: {len(rows)}")

    log_rows = []
    for i, row in enumerate(rows, 1):
        accession = row["accession"]
        fna_path = row["fna_path"]
        genome_outdir = os.path.join(outdir, accession)

        if already_done(accession, outdir):
            print(f"[{i}/{len(rows)}] {accession}: ya procesado, se omite")
            log_rows.append({"accession": accession, "status": "skipped_already_done", "stderr_tail": ""})
            continue

        os.makedirs(genome_outdir, exist_ok=True)
        cmd = [
            platon_bin,
            "--db", platon_db,
            "--output", genome_outdir,
            "--threads", str(threads),
            "--mode", mode,
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
            print(f"No se encontro el ejecutable de Platon ('{platon_bin}'). "
                  f"Verifica instalacion/activacion del entorno conda.", file=sys.stderr)
            sys.exit(1)

        log_rows.append({"accession": accession, "status": status, "stderr_tail": stderr_tail})

    with open(log, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["accession", "status", "stderr_tail"])
        writer.writeheader()
        writer.writerows(log_rows)

    n_ok = sum(1 for r in log_rows if r["status"] == "ok")
    n_skip = sum(1 for r in log_rows if r["status"] == "skipped_already_done")
    n_fail = len(log_rows) - n_ok - n_skip
    print(f"\nCompletados OK: {n_ok} | Omitidos (ya hechos): {n_skip} | Fallidos: {n_fail}")
    print(f"Log detallado en: {log}")


if __name__ == "__main__":
    main()

