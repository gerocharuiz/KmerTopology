
import json
from pathlib import Path

# Directorio donde está el proyecto y se guardan las configuraciones
CONFIGS_DIR = Path(
    "/home/jupyter-gerardorocha/proyectos/KmerTopology/configs"
)

# Directorio base de los datos y resultados
BASE = Path("/files2/gerardo/KmerTopology")

# Configuraciones: (kmer_size, step_size, max_step)
experimentos = [
    (2, 5, 50),
    (2, 4, 48),
    (2, 2, 20),
    (2, 10, 100),
]

for k, r, max_step in experimentos:

    nombre = f"kmer_topology_k{k}_r{r}_max{max_step}"

    # Directorios de configuración, datos y resultados
    config_dir = CONFIGS_DIR / nombre
    data_dir = BASE / "data" / "processed" / nombre
    figures_dir = BASE / "results" / "figures" / nombre
    knn_dir = BASE / "results" / "knn" / nombre / "1_knn"

    # Crear los directorios necesarios
    config_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    knn_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # config_04_vectores.json
    # --------------------------------------------------
    config_04 = {
        "plasmid_manifest": str(
            BASE / "data" / "processed" /
            "selected_genomes" / "plasmid_manifest.csv"
        ),
        "n_genomes": 300,
        "seed": 42,
        "names_out": str(
            data_dir / "escherichia_plasmid.csv"
        ),
        "mat_B": str(
            data_dir / "vectors_topB_escherichia.csv"
        ),
        "mat_E": str(
            data_dir / "vectors_topE_escherichia.csv"
        ),
        "kmer_size": k,
        "step_size": r,
        "max_step": max_step
    }

    # --------------------------------------------------
    # config_05_distancia.json
    # --------------------------------------------------
    config_05 = {
        "mat_B": str(
            data_dir / "vectors_topB_escherichia.csv"
        ),
        "mat_E": str(
            data_dir / "vectors_topE_escherichia.csv"
        ),
        "mat_DB": str(
            data_dir / "distance_B_escherichia.csv"
        ),
        "mat_DE": str(
            data_dir / "distance_E_escherichia.csv"
        ),
        "nombres_csv": str(
            data_dir / "escherichia_plasmid.csv"
        )
    }

    # --------------------------------------------------
    # config_06_tree.json
    # --------------------------------------------------
    config_06 = {
        "mat_DB": str(
            data_dir / "distance_B_escherichia.csv"
        ),
        "mat_DE": str(
            data_dir / "distance_E_escherichia.csv"
        ),
        "nombres_csv": str(
            data_dir / "escherichia_plasmid.csv"
        ),
        "TreeB": str(
            figures_dir / "tree_B_escherichia.png"
        ),
        "TreeE": str(
            figures_dir / "tree_E_escherichia.png"
        )
    }

    # Diccionario con los archivos que se generarán
    configuraciones = {
        "config_04_vectores.json": config_04,
        "config_05_distancia.json": config_05,
        "config_06_tree.json": config_06
    }

    # Guardar los archivos JSON
    for archivo, contenido in configuraciones.items():

        ruta = config_dir / archivo

        with ruta.open("w", encoding="utf-8") as f:
            json.dump(
                contenido,
                f,
                indent=4,
                ensure_ascii=False
            )
            f.write("\n")

        print(f"Creado: {ruta}")

    print(f"Experimento terminado: {nombre}\n")

print("Todas las configuraciones se generaron correctamente.")
