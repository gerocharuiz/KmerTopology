"""
pipeline_common.py

Utilidades compartidas entre run_pipeline_knn.py y run_pipeline_pca.py:
- Cargar la matriz de vectores topologicos (salida de run_pipeline.py: mat_B / mat_E).
- Cargar los accessions en el mismo orden que las filas de esa matriz
  (salida "names_out" de run_pipeline.py).
- Cargar el fenotipo (resistente/susceptible) de cada accession desde
  CUALQUIER manifiesto que tenga esa info -- p.ej. selected_genomes.csv
  (script 01) o plasmid_manifest.csv (script 03), ambos tienen columna
  "phenotype" y "accession".
- Alinear las tres cosas (vector, accession, fenotipo), avisando de
  accessions sin fenotipo o sin vector.
- Reduccion PCA reutilizable y una funcion de scatter 2D generica para
  las imagenes de ambos pipelines.

Autor: Gerardo Rocha Ruiz Jr
"""
from pathlib import Path

import numpy as np
import pandas as pd


def load_vectors(vectors_csv):
    """mat_B / mat_E generados por run_pipeline.py (pd.DataFrame(mat).to_csv(index=False))."""
    df = pd.read_csv(vectors_csv, header=None)
    return df.to_numpy()


def load_names(names_csv, id_column="name"):
    """CSV de accessions en el mismo orden que las filas de la matriz de vectores
    (el "names_out" de run_pipeline.py, o cualquier csv con esa columna)."""
    df = pd.read_csv(names_csv)
    return df[id_column].astype(str).tolist()


def load_label_lookup(labels_manifest, id_column="accession", label_column="phenotype"):
    """Regresa un dict accession -> fenotipo, a partir de CUALQUIER manifiesto que
    tenga esas dos columnas (selected_genomes.csv de 01, o plasmid_manifest.csv de 03)."""
    df = pd.read_csv(labels_manifest, dtype=str)
    df = df.dropna(subset=[id_column, label_column])
    return dict(zip(df[id_column].astype(str), df[label_column].astype(str)))


def assemble_dataset(vectors, ids, label_lookup):
    """
    Alinea vectores <-> accessions <-> fenotipo.

    - vectors: np.ndarray (n_filas, n_features), en el mismo orden que `ids`.
    - ids: lista de accessions, mismo orden y misma longitud que `vectors`.
    - label_lookup: dict accession -> fenotipo.

    Regresa (X, ids_usados, labels_usados, ids_sin_label) -- las filas sin
    fenotipo conocido se excluyen del dataset pero se reportan.
    """
    if len(ids) != vectors.shape[0]:
        raise ValueError(
            f"El numero de accessions ({len(ids)}) no coincide con el numero de "
            f"filas de la matriz de vectores ({vectors.shape[0]}). "
            f"Verifica que 'names_csv' venga del mismo run_pipeline.py que 'vectors'."
        )

    keep_idx = []
    ids_sin_label = []
    for i, accession in enumerate(ids):
        if accession in label_lookup:
            keep_idx.append(i)
        else:
            ids_sin_label.append(accession)

    X = vectors[keep_idx]
    ids_usados = [ids[i] for i in keep_idx]
    labels_usados = np.array([label_lookup[a] for a in ids_usados])
    return X, ids_usados, labels_usados, ids_sin_label


def pca_projection(X, n_components=2, seed=None):
    """PCA de sklearn, regresa (coords, explained_variance_ratio, pca_obj)."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, random_state=seed)
    coords = pca.fit_transform(X)
    return coords, pca.explained_variance_ratio_, pca


def label_colors(labels):
    """Mapa fenotipo -> color, estable y determinista (ordenado alfabeticamente).
    Con 2 clases usa rojo/azul (resistente suele venir primero alfabeticamente
    en datasets tipo R/S o Resistant/Susceptible); con mas clases usa un colormap."""
    unique_labels = sorted(set(labels))
    if len(unique_labels) == 2:
        palette = ["#d62728", "#1f77b4"]  # rojo, azul
    else:
        import matplotlib.cm as cm
        cmap = cm.get_cmap("tab10", max(len(unique_labels), 3))
        palette = [cmap(i) for i in range(len(unique_labels))]
    return {lab: palette[i % len(palette)] for i, lab in enumerate(unique_labels)}


def plot_label_scatter(coords, labels, out_path, title,
                        ids=None, annotate=False,
                        markers_by=None, marker_map=None,
                        xlabel="Componente 1", ylabel="Componente 2"):
    """
    Scatter 2D coloreado por fenotipo (resistente/susceptible/etc).

    - coords: array (n, 2).
    - labels: array/lista de fenotipos, longitud n, usada para el color.
    - markers_by / marker_map: opcional, para usar OTRA variable categorica
      (p.ej. train/test, o correcto/incorrecto en KNN) como forma del marcador.
    - annotate: si True, escribe el accession junto a cada punto (util solo
      con pocos puntos).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = label_colors(labels)
    fig, ax = plt.subplots(figsize=(8, 6))

    if markers_by is not None:
        marker_map = marker_map or {}
        default_marker = "o"
        categories = sorted(set(markers_by))
        for cat in categories:
            marker = marker_map.get(cat, default_marker)
            mask = np.array([m == cat for m in markers_by])
            for lab in sorted(set(labels)):
                lab_mask = mask & (np.array(labels) == lab)
                if lab_mask.any():
                    ax.scatter(
                        coords[lab_mask, 0], coords[lab_mask, 1],
                        c=[colors[lab]], marker=marker,
                        label=f"{lab} ({cat})", edgecolor="black", linewidths=0.4, s=60,
                    )
    else:
        for lab in sorted(set(labels)):
            mask = np.array(labels) == lab
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=[colors[lab]], label=str(lab),
                edgecolor="black", linewidths=0.4, s=60,
            )

    if annotate and ids is not None:
        for (x, y), acc in zip(coords, ids):
            ax.annotate(acc, (x, y), fontsize=6, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
