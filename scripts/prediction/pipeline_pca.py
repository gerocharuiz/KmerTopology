"""
run_pipeline_pca.py

Aplica PCA a los vectores topologicos generados por run_pipeline.py
(mat_B o mat_E) y genera una imagen de dispersion donde cada punto es un
genoma, coloreado por su fenotipo (resistente/susceptible), tomando ese
fenotipo de CUALQUIER manifiesto que ya tengas (selected_genomes.csv de 01,
o plasmid_manifest.csv de 03).

Que genera:
    - Imagen de dispersion PC1 vs PC2 coloreada por fenotipo (con opcion de
      anotar el accession de cada punto).
    - CSV con las coordenadas PCA de cada genoma + su accession + fenotipo,
      por si quieres graficarlo distinto o alimentarlo a otro analisis.
    - JSON con la varianza explicada por componente.

Requiere: pandas, numpy, scikit-learn, matplotlib.

Uso:
    python run_pipeline_pca.py config_pca.json

Config de ejemplo:
{
    "vectors": "data/processed/vectors_topB_escherichia.csv",
    "names_csv": "data/processed/escherichia_plasmid_200.csv",
    "labels_manifest": "data/processed/plasmid_manifest.csv",
    "id_column": "accession",
    "label_column": "phenotype",
    "n_components": 2,
    "scale_features": true,
    "annotate_points": false,
    "seed": 42,
    "out_dir": "results/pca_escherichia",
    "prefix": "pca_escherichia"
}

Autor: Gerardo Rocha Ruiz Jr
"""
import json
import sys
from pathlib import Path

import pandas as pd

from pipeline_common import (
    load_vectors, load_names, load_label_lookup, assemble_dataset,
    pca_projection, plot_label_scatter,
)


def main():
    if len(sys.argv) != 2:
        print("Uso: python run_pipeline_pca.py config.json", file=sys.stderr)
        sys.exit(1)

    config_file = Path(sys.argv[1])
    with open(config_file, encoding="utf-8") as f:
        config = json.load(f)

    vectors_path = config["vectors"]
    names_csv = config["names_csv"]
    labels_manifest = config["labels_manifest"]
    id_column = config.get("id_column", "accession")
    label_column = config.get("label_column", "phenotype")
    names_id_column = config.get("names_id_column", "name")

    n_components = config.get("n_components", 2)
    scale_features = config.get("scale_features", True)
    annotate_points = config.get("annotate_points", False)
    seed = config.get("seed", 42)

    out_dir = Path(config.get("out_dir", "results/pca"))
    prefix = config.get("prefix", "pca")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Cargar y alinear datos ---
    vectors = load_vectors(vectors_path)
    ids = load_names(names_csv, id_column=names_id_column)
    label_lookup = load_label_lookup(labels_manifest, id_column=id_column, label_column=label_column)

    X, ids_usados, y, ids_sin_label = assemble_dataset(vectors, ids, label_lookup)
    print(f"Genomas con vector y fenotipo: {len(ids_usados)}")
    if ids_sin_label:
        print(f"Genomas sin fenotipo en el manifiesto (excluidos): {len(ids_sin_label)}")

    if scale_features:
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)

    coords, evr, _ = pca_projection(X, n_components=n_components, seed=seed)

    # --- CSV con coordenadas + accession + fenotipo ---
    coord_cols = {f"PC{i + 1}": coords[:, i] for i in range(coords.shape[1])}
    df_out = pd.DataFrame({id_column: ids_usados, label_column: y, **coord_cols})
    coords_csv = out_dir / f"{prefix}_coords.csv"
    df_out.to_csv(coords_csv, index=False)

    # --- JSON con varianza explicada ---
    variance_out = {
        "n_components": n_components,
        "scale_features": scale_features,
        "explained_variance_ratio": evr.tolist(),
        "explained_variance_ratio_cumsum": evr.cumsum().tolist(),
        "genomas_sin_fenotipo": ids_sin_label,
    }
    variance_json = out_dir / f"{prefix}_explained_variance.json"
    with open(variance_json, "w", encoding="utf-8") as f:
        json.dump(variance_out, f, indent=2, ensure_ascii=False)

    # --- Imagen: dispersion PC1 vs PC2, coloreada por fenotipo ---
    scatter_path = out_dir / f"{prefix}_scatter.png"
    plot_label_scatter(
        coords[:, :2], y, scatter_path,
        title=f"PCA de vectores topologicos coloreado por fenotipo\n"
              f"var. explicada: PC1={evr[0]:.1%}, PC2={evr[1]:.1%}"
              + (f", PC3={evr[2]:.1%}" if n_components >= 3 else ""),
        ids=ids_usados, annotate=annotate_points,
        xlabel=f"PC1 ({evr[0]:.1%})", ylabel=f"PC2 ({evr[1]:.1%})",
    )

    print(f"Varianza explicada: {[f'{v:.1%}' for v in evr]}")
    print(f"\nCoordenadas CSV:  {coords_csv}")
    print(f"Varianza JSON:    {variance_json}")
    print(f"Dispersion PCA:   {scatter_path}")


if __name__ == "__main__":
    main()