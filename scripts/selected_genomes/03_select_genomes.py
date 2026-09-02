#!/usr/bin/env python3
"""
03_build_plasmid_manifest.py

Recorre la salida de Platon (02_run_platon_batch.py) y arma un manifiesto
final: por cada accession, cuantos contigs plasmidicos encontro Platon,
su longitud total, y la ruta al fasta que le vas a pasar a tu metodo de
vector topologico de k-mers (KmerTopology).

Casos por genoma:
    - no_plasmid:     Platon no encontro ningun contig plasmidico -> se
                       excluye del manifiesto final (no hay nada que vectorizar)
    - single_contig:  un solo contig plasmidico -> se usa tal cual
    - multi_contig:   mas de un contig plasmidico -> por default se
                       conservan TODOS en un multi-fasta (algunos metodos de
                       k-mers aceptan multi-fasta por genoma); si prefieres
                       quedarte solo con el contig plasmidico mas largo
                       (heuristica de "plasmido probablemente mas completo"),
                       usa "largest_only": true en el config

El fasta resultante para cada accession se copia/escribe en
<final_dir>/<accession>.plasmid.fasta

Uso (config JSON, mismo estilo que run_pipeline.py / run_pipeline2.py / run_pipeline3.py):
    python 03_build_plasmid_manifest.py config_manifest.json

Donde config_manifest.json luce asi (solo "manifest" y "platon_outdir" son
obligatorias):
{
    "manifest": "selected_genomes.csv",
    "platon_outdir": "platon_out",
    "final_dir": "plasmid_contigs",
    "out": "plasmid_manifest.csv",
    "largest_only": false
}

Nota: si dejas "largest_only": false aqui, run_pipeline.py (script 1 de tu
pipeline de KmerTopology) puede encargarse de quedarse con el contig mas
largo de cada fasta el momento de vectorizar -- ver su modo
"plasmid_manifest" en el config.
"""
import csv
import json
import os
import sys
from pathlib import Path


def parse_fasta(path):
    """Parser minimo de FASTA: regresa lista de (header, seq) sin depender de Biopython."""
    records = []
    header = None
    seq_chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_chunks)))
                header = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header is not None:
            records.append((header, "".join(seq_chunks)))
    return records


def write_fasta(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for header, seq in records:
            f.write(f">{header}\n")
            for i in range(0, len(seq), 70):
                f.write(seq[i:i + 70] + "\n")


def main():
    if len(sys.argv) != 2:
        print("Uso: python 03_build_plasmid_manifest.py config.json", file=sys.stderr)
        sys.exit(1)

    config_file = Path(sys.argv[1])
    with open(config_file, encoding="utf-8") as f:
        config = json.load(f)

    manifest = config["manifest"]
    platon_outdir = config["platon_outdir"]
    final_dir = config.get("final_dir", "plasmid_contigs")
    out = config.get("out", "plasmid_manifest.csv")
    largest_only = bool(config.get("largest_only", False))

    os.makedirs(final_dir, exist_ok=True)

    with open(manifest, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["found"] in ("True", "true", "1")]

    out_rows = []
    for row in rows:
        accession = row["accession"]
        plasmid_fasta = os.path.join(platon_outdir, accession, f"{accession}.plasmid.fasta")

        if not os.path.isfile(plasmid_fasta) or os.path.getsize(plasmid_fasta) == 0:
            out_rows.append({
                "accession": accession, "phenotype": row["phenotype"],
                "n_plasmid_contigs": 0, "total_plasmid_bp": 0,
                "status": "no_plasmid", "final_fasta_path": "",
            })
            continue

        records = parse_fasta(plasmid_fasta)
        n_contigs = len(records)
        total_bp = sum(len(seq) for _, seq in records)

        if n_contigs == 0:
            status = "no_plasmid"
            final_path = ""
        elif n_contigs == 1:
            status = "single_contig"
            final_path = os.path.join(final_dir, f"{accession}.plasmid.fasta")
            write_fasta(records, final_path)
        else:
            status = "multi_contig"
            final_path = os.path.join(final_dir, f"{accession}.plasmid.fasta")
            if largest_only:
                largest = max(records, key=lambda r: len(r[1]))
                write_fasta([largest], final_path)
                total_bp = len(largest[1])
                n_contigs = 1
                status = "multi_contig_largest_kept"
            else:
                write_fasta(records, final_path)

        out_rows.append({
            "accession": accession, "phenotype": row["phenotype"],
            "n_plasmid_contigs": n_contigs, "total_plasmid_bp": total_bp,
            "status": status, "final_fasta_path": final_path,
        })

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["accession", "phenotype", "n_plasmid_contigs",
                           "total_plasmid_bp", "status", "final_fasta_path"])
        writer.writeheader()
        writer.writerows(out_rows)

    n_none = sum(1 for r in out_rows if r["status"] == "no_plasmid")
    n_single = sum(1 for r in out_rows if r["status"] == "single_contig")
    n_multi = sum(1 for r in out_rows if r["status"] in ("multi_contig", "multi_contig_largest_kept"))
    print(f"Genomas sin plasmido detectado: {n_none}")
    print(f"Genomas con 1 contig plasmidico: {n_single}")
    print(f"Genomas con >1 contig plasmidico: {n_multi}")
    print(f"Manifiesto final: {out}")
    print(f"Fastas listos para KmerTopology en: {final_dir}/")


if __name__ == "__main__":
    main()
