"""
run_pipeline_knn.py

Clasificador KNN de resistencia/susceptibilidad a partir de los vectores
topologicos generados por run_pipeline.py (mat_B o mat_E), usando el
fenotipo de CUALQUIER manifiesto que ya tengas (selected_genomes.csv de 01,
o plasmid_manifest.csv de 03 -- ambos traen columna "phenotype" +
"accession").

Tiene dos modos, segun la config:

1) Un solo split train/test (comportamiento original, default si no pones
   "cv_folds" en el config).

2) Validacion cruzada ("cv_folds": k en el config): parte los datos en k
   formas distintas de armar el conjunto de test (StratifiedKFold, cada
   genoma cae en el test exactamente una vez a lo largo de las k vueltas).
   Reporta la MISMA informacion que el modo (1) para cada particion
   (accuracy, classification report, matriz de confusion), y ADEMAS un
   bloque "aggregate" con la union de las predicciones de las k particiones
   (cada genoma aparece una sola vez) y sus metricas globales, mas el
   promedio +/- desviacion estandar del accuracy entre particiones.
   Las imagenes NO se generan por cada particion -- solo una vez, usando
   por default la union de todas las predicciones ("image_source":
   "aggregate"), o una particion especifica si pones
   "image_source": "fold" y "image_fold_index": <indice>.

Que genera (en ambos modos):
    - Metricas en JSON.
    - Predicciones por genoma en CSV (con columna "fold" en modo cv).
    - Imagen de la matriz de confusion (una sola, ver arriba).
    - Imagen de dispersion (proyeccion PCA 2D de los vectores) coloreada
      por fenotipo real, marcando ademas si el modelo acerto o no (y en
      modo de un solo split, si es train o test) -- para "ver" la
      resistencia directamente sobre el espacio de vectores.

Requiere: pandas, numpy, scikit-learn, matplotlib.

Uso:
    python run_pipeline_knn.py config_knn.json

Config de ejemplo (un solo split, como antes):
{
    "vectors": "data/processed/vectors_topB_escherichia.csv",
    "names_csv": "data/processed/escherichia_plasmid_200.csv",
    "labels_manifest": "data/processed/plasmid_manifest.csv",
    "n_neighbors": 5,
    "metric": "euclidean",
    "scale_features": true,
    "test_size": 0.25,
    "seed": 42,
    "out_dir": "results/knn_escherichia",
    "prefix": "knn_escherichia",
    "annotate_points": false
}

Config de ejemplo (validacion cruzada, 5 formas de armar el test):
{
    "vectors": "data/processed/vectors_topB_escherichia.csv",
    "names_csv": "data/processed/escherichia_plasmid_200.csv",
    "labels_manifest": "data/processed/plasmid_manifest.csv",
    "n_neighbors": 5,
    "metric": "euclidean",
    "scale_features": true,
    "cv_folds": 5,
    "shuffle_folds": true,
    "seed": 42,
    "image_source": "aggregate",
    "out_dir": "results/knn_escherichia_cv",
    "prefix": "knn_escherichia_cv",
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


def fit_predict(X_train, X_test, y_train, n_neighbors, metric, scale_features):
    """Escala (opcional), entrena KNN con el train y predice el test."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier

    if scale_features:
        scaler = StandardScaler()
        X_train_fit = scaler.fit_transform(X_train)
        X_test_fit = scaler.transform(X_test)
    else:
        X_train_fit, X_test_fit = X_train, X_test

    clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    clf.fit(X_train_fit, y_train)
    return clf.predict(X_test_fit)


def fold_report(y_test, y_pred, labels_sorted, n_train, n_test, fold=None):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, labels=labels_sorted, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    out = {
        "n_train": n_train,
        "n_test": n_test,
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    if fold is not None:
        out = {"fold": fold, **out}
    return out, acc, cm


def save_confusion_matrix_image(cm, labels_sorted, title, out_path):
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
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


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
    seed = config.get("seed", 42)

    cv_folds = config.get("cv_folds")  # None -> modo de un solo split (compatibilidad hacia atras)
    test_size = config.get("test_size", 0.25)
    shuffle_folds = config.get("shuffle_folds", True)
    image_source = config.get("image_source", "aggregate")  # "aggregate" o "fold"
    image_fold_index = config.get("image_fold_index", 0)

    out_dir = Path(config.get("out_dir", "results/knn"))
    prefix = config.get("prefix", "knn")
    annotate_points = config.get("annotate_points", False)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Cargar y alinear datos ---
    vectors = load_vectors(vectors_path)
    ids = load_names(names_csv, id_column=names_id_column)
    label_lookup = load_label_lookup(labels_manifest, id_column=id_column, label_column=label_column)

    X, ids_usados, y, ids_sin_label = assemble_dataset(vectors, ids, label_lookup)
    print(f"Genomas con vector y fenotipo: {len(ids_usados)}")
    if ids_sin_label:
        print(f"Genomas sin fenotipo en el manifiesto (excluidos): {len(ids_sin_label)}")

    labels_sorted = sorted(set(y))
    cm_path = out_dir / f"{prefix}_confusion_matrix.png"
    scatter_path = out_dir / f"{prefix}_pca_resistencia.png"

    if not cv_folds or cv_folds <= 1:
        # ============== MODO ORIGINAL: un solo split train/test ==============
        from sklearn.model_selection import train_test_split

        idx = np.arange(len(ids_usados))
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, idx, test_size=test_size, random_state=seed, stratify=y,
        )
        y_pred = fit_predict(X_train, X_test, y_train, n_neighbors, metric, scale_features)

        report, acc, cm = fold_report(y_test, y_pred, labels_sorted, len(y_train), len(y_test))
        print(f"Accuracy en test: {acc:.4f}")

        report_out = {
            "mode": "single_split",
            "n_neighbors": n_neighbors,
            "metric": metric,
            "scale_features": scale_features,
            "test_size": test_size,
            "seed": seed,
            "confusion_matrix_labels": labels_sorted,
            "genomas_sin_fenotipo": ids_sin_label,
            **report,
        }

        ids_test = [ids_usados[i] for i in idx_test]
        df_pred = pd.DataFrame({
            id_column: ids_test,
            "fenotipo_real": y_test,
            "fenotipo_predicho": y_pred,
            "correcto": y_test == y_pred,
        })

        # Imagen: matriz de confusion de este (unico) split
        save_confusion_matrix_image(
            cm, labels_sorted,
            f"Matriz de confusion (KNN, k={n_neighbors})\naccuracy={acc:.3f}",
            cm_path,
        )

        # Imagen: dispersion PCA, marcando train / test-acierto / test-error
        from sklearn.preprocessing import StandardScaler
        X_all_scaled = StandardScaler().fit_transform(X) if scale_features else X
        coords, evr, _ = pca_projection(X_all_scaled, n_components=2, seed=seed)

        marker_group = np.array(["train"] * len(ids_usados), dtype=object)
        for pos, i in enumerate(idx_test):
            marker_group[i] = "test_correcto" if (y_test[pos] == y_pred[pos]) else "test_error"

        plot_label_scatter(
            coords, y, scatter_path,
            title=f"Vectores topologicos (PCA 2D) coloreados por fenotipo\n"
                  f"var. explicada: PC1={evr[0]:.1%}, PC2={evr[1]:.1%}",
            ids=ids_usados, annotate=annotate_points,
            markers_by=marker_group,
            marker_map={"train": "o", "test_correcto": "^", "test_error": "x"},
        )

    else:
        # ============== MODO VALIDACION CRUZADA: k formas de armar el test ==============
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(
            n_splits=cv_folds, shuffle=shuffle_folds,
            random_state=seed if shuffle_folds else None,
        )

        fold_reports = []
        pred_rows = []
        # se llenan a lo largo de las k vueltas: cada genoma cae en el test
        # exactamente una vez, asi que al final cubren TODO el dataset
        y_true_all = np.empty(len(y), dtype=object)
        y_pred_all = np.empty(len(y), dtype=object)
        fold_image_data = None

        for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            y_pred = fit_predict(X_train, X_test, y_train, n_neighbors, metric, scale_features)

            report, acc, _ = fold_report(
                y_test, y_pred, labels_sorted, len(y_train), len(y_test), fold=fold_i,
            )
            fold_reports.append(report)
            print(f"Fold {fold_i}: accuracy = {acc:.4f} (n_test={len(y_test)})")

            y_true_all[test_idx] = y_test
            y_pred_all[test_idx] = y_pred

            for pos, i in enumerate(test_idx):
                pred_rows.append({
                    id_column: ids_usados[i],
                    "fold": fold_i,
                    "fenotipo_real": y_test[pos],
                    "fenotipo_predicho": y_pred[pos],
                    "correcto": y_test[pos] == y_pred[pos],
                })

            if image_source == "fold" and fold_i == image_fold_index:
                fold_image_data = dict(
                    idx_test=test_idx, y_test=y_test, y_pred=y_pred, acc=acc,
                )

        # --- Union de todas las particiones (cada genoma aparece una vez) ---
        aggregate_report, aggregate_acc, aggregate_cm = fold_report(
            y_true_all, y_pred_all, labels_sorted, n_train=None, n_test=len(y),
        )
        del aggregate_report["n_train"]  # no aplica: cada genoma paso por train en k-1 folds

        fold_accuracies = [r["accuracy"] for r in fold_reports]
        aggregate_report["accuracy_por_fold"] = fold_accuracies
        aggregate_report["accuracy_media_folds"] = float(np.mean(fold_accuracies))
        aggregate_report["accuracy_std_folds"] = float(np.std(fold_accuracies))

        print(f"\nAccuracy agregado (union de las {cv_folds} particiones): {aggregate_acc:.4f}")
        print(f"Accuracy promedio +/- std entre particiones: "
              f"{np.mean(fold_accuracies):.4f} +/- {np.std(fold_accuracies):.4f}")

        report_out = {
            "mode": "cross_validation",
            "cv_folds": cv_folds,
            "shuffle_folds": shuffle_folds,
            "n_neighbors": n_neighbors,
            "metric": metric,
            "scale_features": scale_features,
            "seed": seed,
            "confusion_matrix_labels": labels_sorted,
            "genomas_sin_fenotipo": ids_sin_label,
            "folds": fold_reports,
            "aggregate": aggregate_report,
        }

        df_pred = pd.DataFrame(pred_rows).sort_values("fold").reset_index(drop=True)

        # --- Imagen (una sola, por default la union de todas las particiones) ---
        if image_source == "fold" and fold_image_data is not None:
            d = fold_image_data
            cm_title = (f"Matriz de confusion (KNN, k={n_neighbors}, fold {image_fold_index}"
                        f" de {cv_folds})\naccuracy={d['acc']:.3f}")
            _, _, cm_img = fold_report(d["y_test"], d["y_pred"], labels_sorted, None, None)
            save_confusion_matrix_image(cm_img, labels_sorted, cm_title, cm_path)

            from sklearn.preprocessing import StandardScaler
            X_all_scaled = StandardScaler().fit_transform(X) if scale_features else X
            coords, evr, _ = pca_projection(X_all_scaled, n_components=2, seed=seed)
            marker_group = np.array(["train"] * len(ids_usados), dtype=object)
            for pos, i in enumerate(d["idx_test"]):
                marker_group[i] = "test_correcto" if (d["y_test"][pos] == d["y_pred"][pos]) else "test_error"
            scatter_title = (f"Vectores topologicos (PCA 2D), fold {image_fold_index} de {cv_folds}\n"
                              f"var. explicada: PC1={evr[0]:.1%}, PC2={evr[1]:.1%}")
        else:
            cm_title = (f"Matriz de confusion (KNN, k={n_neighbors}) -- union de "
                        f"{cv_folds} particiones\naccuracy agregado={aggregate_acc:.3f}")
            save_confusion_matrix_image(aggregate_cm, labels_sorted, cm_title, cm_path)

            from sklearn.preprocessing import StandardScaler
            X_all_scaled = StandardScaler().fit_transform(X) if scale_features else X
            coords, evr, _ = pca_projection(X_all_scaled, n_components=2, seed=seed)
            marker_group = np.where(y_true_all == y_pred_all, "correcto", "error")
            scatter_title = (f"Vectores topologicos (PCA 2D) -- predicciones agregadas de "
                              f"{cv_folds} particiones\nvar. explicada: PC1={evr[0]:.1%}, "
                              f"PC2={evr[1]:.1%}")

        plot_label_scatter(
            coords, y, scatter_path,
            title=scatter_title,
            ids=ids_usados, annotate=annotate_points,
            markers_by=marker_group,
            marker_map={"train": "o", "test_correcto": "^", "test_error": "x",
                        "correcto": "^", "error": "x"},
        )

    # --- Guardar reporte + predicciones (comun a ambos modos) ---
    report_path = out_dir / f"{prefix}_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_out, f, indent=2, ensure_ascii=False)

    predictions_path = out_dir / f"{prefix}_predictions.csv"
    df_pred.to_csv(predictions_path, index=False)

    print(f"\nReporte JSON:            {report_path}")
    print(f"Predicciones CSV:        {predictions_path}")
    print(f"Matriz de confusion:     {cm_path}")
    print(f"Dispersion PCA/fenotipo: {scatter_path}")


if __name__ == "__main__":
    main()
