import numpy as np
import pandas as pd
import re
from scipy.sparse import csr_matrix


# ============================================================
# 1. Archivos
# ============================================================

BASE = "./data/raw/metadata"

metadata_file = f"{BASE}/pangenome.metadata.csv"
escherichia_file = f"{BASE}/escherichia.csv"
npz_file = f"{BASE}/pangenome.npz"
features_file = f"{BASE}/pangenome.features.txt"

output_file = f"{BASE}/escherichia_plasmid_CARD.csv"


# ============================================================
# 2. Leer metadata
# ============================================================

metadata = pd.read_csv(metadata_file)
escherichia = pd.read_csv(escherichia_file)

print("Genomas en pangenome.metadata:",
      len(metadata))

print("Genomas en escherichia.csv:",
      len(escherichia))


# ============================================================
# 3. Cargar pangenome.npz
# ============================================================

data = np.load(npz_file)

matrix = csr_matrix(
    (
        data["data"],
        data["indices"],
        data["indptr"]
    ),
    shape=tuple(data["shape"])
)

print("Matriz del pangenoma:", matrix.shape)


# ============================================================
# 4. Cargar features
# ============================================================

with open(features_file) as f:
    features = [line.strip() for line in f]

print("Número de features:", len(features))


# ============================================================
# 5. Crear accession -> fila
# ============================================================

accession_to_row = {
    accession: i
    for i, accession in enumerate(metadata["accession"])
}


# ============================================================
# 6. Procesar todos los genomas de Escherichia
# ============================================================

resultados = []

no_encontrados = []


for _, genome in escherichia.iterrows():

    accession = genome["accession"]

    # --------------------------------------------------------
    # Comprobar que existe en el pangenoma
    # --------------------------------------------------------

    if accession not in accession_to_row:

        no_encontrados.append(accession)

        continue

    fila = accession_to_row[accession]

    # --------------------------------------------------------
    # Obtener features presentes
    # --------------------------------------------------------

    feature_indices = matrix[fila].indices

    # --------------------------------------------------------
    # Obtener solamente P_A...
    #
    # P = localizado en plásmido
    # A = anotado mediante CARD
    # --------------------------------------------------------

    for i in feature_indices:

        feature = features[i]

        if not feature.startswith("P_A"):
            continue

        # ----------------------------------------------------
        # Extraer ARO
        # ----------------------------------------------------

        match = re.match(
            r"P_A(\d+)_",
            feature
        )

        if not match:
            continue

        aro = int(match.group(1))

        # ----------------------------------------------------
        # Guardar información del genoma + feature
        # ----------------------------------------------------

        resultados.append({

            # Información del genoma
            "accession": accession,
            "biosample": genome["biosample"],
            "species": genome["species"],

            # Antibiótico y fenotipo
            "antibiotic": genome["antibiotic"],
            "phenotype": genome["phenotype"],

            # Información adicional
            "mic": genome["mic"],
            "adjusted_mic": genome["adjusted_mic"],
            "adjusted_phenotype": genome["adjusted_phenotype"],

            # Feature
            "feature": feature,

            # CARD
            "aro": aro,

            # Según la notación del dataset:
            # P_A... -> localizado en plásmido
            "location": "plasmid"

        })


# ============================================================
# 7. Crear DataFrame
# ============================================================

resultado = pd.DataFrame(resultados)


# ============================================================
# 8. Guardar
# ============================================================

resultado.to_csv(
    output_file,
    index=False
)


# ============================================================
# 9. Mostrar resultados
# ============================================================

print("\n========================================")
print("RESULTADOS")
print("========================================")

print("Número de candidatos:", len(resultado))

print("\nPrimeras filas:")

print(
    resultado[
        [
            "accession",
            "antibiotic",
            "phenotype",
            "feature",
            "aro"
        ]
    ].head(20)
)


# ============================================================
# 10. Genomas que no fueron encontrados
# ============================================================

if no_encontrados:

    print("\n========================================")
    print("ACCESSIONS NO ENCONTRADOS")
    print("========================================")

    for accession in no_encontrados:
        print(accession)

else:

    print("\nTodos los accession fueron encontrados.")


print("\nArchivo guardado en:")
print(output_file)