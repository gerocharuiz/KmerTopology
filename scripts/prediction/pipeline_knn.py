"""
run_pipeline_knn.py

Clasificador KNN de resistencia/susceptibilidad a partir de los vectores
topologicos generados por run_pipeline.py (mat_B o mat_E), usando el
fenotipo de CUALQUIER manifiesto que ya tengas (selected_genomes.csv de 01,
o plasmid_manifest.csv de 03 -- ambos traen columna "phenotype" +
"accession").

Que genera:
    - Metricas (accuracy, classification_report, matriz de confusion) en JSON.
    - Predicciones por genoma (accession, fenotipo real, fenotipo predicho,
      si acerto) en CSV.
    - Imagen de la matriz de confusion.
    - Imagen de dispersion (proyeccion PCA 2D de los vectores) coloreada
      por fenotipo real, marcando ademas si cada punto es de train/test y
      si el modelo lo clasifico bien o mal -- para "ver" la resistencia
      directamente sobre el espacio de vectores.

Requiere: pandas, numpy, scikit-learn, matplotlib.

Uso:
    python run_pipeline_knn.py config_knn.json

Config de ejemplo:
{
    "vectors": "data/processed/vectors_topB_escherichia.csv",
    "names_csv": "data/processed/escherichia_plasmid_200.csv",
    "labels_manifest": "data/processed/plasmid_manifest.csv",
    "id_column": "accession",
    "label_column": "phenotype",
    "n_neighbors": 5,
    "metric": "euclidean",
    "scale_features": true,
    "test_size": 0.25,
    "seed": 42,
    "out_dir": "results/knn_escherichia",
    "prefix": "knn_escherichia",
    "annotate_points": false
}

Autor: Gerardo Rocha Ruiz Jr
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import (
    load_vectors, load_names, load_label_lookup, assemble_dataset,
    pca_projection, plot_label_scatter,
)


def main():
    if len(sys.argv) != 2:
        print("Uso: python run_pipeline_knn.py config.json", file=sys.stderr)
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

    n_neighbors = config.get("n_neighbors", 5)
    metric = config.get("metric", "euclidean")
    scale_features = config.get("scale_features", True)
    test_size = config.get("test_size", 0.25)
    seed = config.get("seed", 42)

    out_dir = Path(config.get("out_dir", "results/knn"))
    prefix = config.get("prefix", "knn")
    annotate_points = config.get("annotate_points", False)
    out_dir.mkdir(parents=True, exist_ok=True)

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
    )

    # --- Cargar y alinear datos ---
    vectors = load_vectors(vectors_path)
    ids = load_names(names_csv, id_column=names_id_column)
    label_lookup = load_label_lookup(labels_manifest, id_column=id_column, label_column=label_column)

    X, ids_usados, y, ids_sin_label = assemble_dataset(vectors, ids, label_lookup)
    print(f"Genomas con vector y fenotipo: {len(ids_usados)}")
    if ids_sin_label:
        print(f"Genomas sin fenotipo en el manifiesto (excluidos): {len(ids_sin_label)}")

    idx = np.arange(len(ids_usados))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, idx, test_size=test_size, random_state=seed, stratify=y,
    )

    if scale_features:
        scaler = StandardScaler()
        X_train_fit = scaler.fit_transform(X_train)
        X_test_fit = scaler.transform(X_test)
    else:
        X_train_fit, X_test_fit = X_train, X_test

    clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    clf.fit(X_train_fit, y_train)
    y_pred = clf.predict(X_test_fit)

    # --- Metricas ---
    labels_sorted = sorted(set(y))
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, labels=labels_sorted, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    print(f"Accuracy en test: {acc:.4f}")
    print(classification_report(y_test, y_pred, labels=labels_sorted))

    report_out = {
        "n_neighbors": n_neighbors,
        "metric": metric,
        "scale_features": scale_features,
        "test_size": test_size,
        "seed": seed,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": labels_sorted,
        "genomas_sin_fenotipo": ids_sin_label,
    }
    with open(out_dir / f"{prefix}_report.json", "w", encoding="utf-8") as f:
        json.dump(report_out, f, indent=2, ensure_ascii=False)

    # --- Predicciones por genoma ---
    ids_test = [ids_usados[i] for i in idx_test]
    df_pred = pd.DataFrame({
        id_column: ids_test,
        "fenotipo_real": y_test,
        "fenotipo_predicho": y_pred,
        "correcto": y_test == y_pred,
    })
    df_pred.to_csv(out_dir / f"{prefix}_predictions.csv", index=False)

    # --- Imagen: matriz de confusion ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels_sorted)))
    ax.set_yticks(range(len(labels_sorted)))
    ax.set_xticklabels(labels_sorted, rotation=45, ha="right")
    ax.set_yticklabels(labels_sorted)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusion (KNN, k={n_neighbors})\naccuracy={acc:.3f}")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    cm_path = out_dir / f"{prefix}_confusion_matrix.png"
    fig.savefig(cm_path, dpi=300)
    plt.close(fig)

    # --- Imagen: dispersion PCA 2D, coloreada por fenotipo real ---
    X_all_scaled = StandardScaler().fit_transform(X) if scale_features else X
    coords, evr, _ = pca_projection(X_all_scaled, n_components=2, seed=seed)

    split_labels = np.array(["train"] * len(ids_usados), dtype=object)
    for i in idx_test:
        split_labels[i] = "test"
    # marcar aciertos/errores solo donde tenemos prediccion (los de test)
    marker_group = split_labels.copy()
    for pos, i in enumerate(idx_test):
        marker_group[i] = "test_correcto" if (y_test[pos] == y_pred[pos]) else "test_error"

    scatter_path = out_dir / f"{prefix}_pca_resistencia.png"
    plot_label_scatter(
        coords, y, scatter_path,
        title=f"Vectores topologicos (PCA 2D) coloreados por fenotipo\n"
              f"var. explicada: PC1={evr[0]:.1%}, PC2={evr[1]:.1%}",
        ids=ids_usados, annotate=annotate_points,
        markers_by=marker_group,
        marker_map={"train": "o", "test_correcto": "^", "test_error": "x"},
    )

    print(f"\nReporte JSON:            {out_dir / f'{prefix}_report.json'}")
    print(f"Predicciones CSV:        {out_dir / f'{prefix}_predictions.csv'}")
    print(f"Matriz de confusion:     {cm_path}")
    print(f"Dispersion PCA/fenotipo: {scatter_path}")


if __name__ == "__main__":
    main()
