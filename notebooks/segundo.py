import json
import pandas as pd


# ============================================================
# 1. Archivos
# ============================================================

CARD_FILE = "./data/raw/metadata/card.json"

INPUT_FILE = (
    "./data/raw/metadata/"
    "escherichia_plasmid_CARD.csv"
)

OUTPUT_FILE = (
    "./data/raw/metadata/"
    "escherichia_plasmid_CARD_annotated.csv"
)

CANDIDATES_FILE = (
    "./data/raw/metadata/"
    "candidatos_plasmid_resistencia.csv"
)


# ============================================================
# 2. Cargar CARD
# ============================================================

print("Cargando CARD...")

with open(CARD_FILE) as f:
    card = json.load(f)

print("Modelos CARD:", len(card))


# ============================================================
# 3. Construir índice ARO -> información CARD
# ============================================================

print("Construyendo índice de ARO...")

aro_index = {}

for model_id, model in card.items():

    # Algunas entradas del JSON no son modelos
    if not isinstance(model, dict):
        continue

    aro = model.get("ARO_accession")

    if aro is None:
        continue

    aro = str(aro)

    categories = model.get("ARO_category", {})

    amr_gene_family = []
    drug_class = []
    resistance_mechanism = []

    for category in categories.values():

        category_type = category.get(
            "category_aro_class_name"
        )

        category_name = category.get(
            "category_aro_name"
        )

        if category_type == "AMR Gene Family":
            amr_gene_family.append(category_name)

        elif category_type == "Drug Class":
            drug_class.append(category_name)

        elif category_type == "Resistance Mechanism":
            resistance_mechanism.append(category_name)

    aro_index[aro] = {
        "aro_name": model.get("ARO_name"),
        "aro_description": model.get("ARO_description"),

        "amr_gene_family": "; ".join(
            sorted(set(amr_gene_family))
        ),

        "drug_class": "; ".join(
            sorted(set(drug_class))
        ),

        "resistance_mechanism": "; ".join(
            sorted(set(resistance_mechanism))
        ),

        "card_model_id": model_id,
        "model_type": model.get("model_type"),
    }


print("AROs indexados:", len(aro_index))


# ============================================================
# 4. Cargar tabla de Escherichia
# ============================================================

print("\nCargando tabla de Escherichia...")

df = pd.read_csv(INPUT_FILE)

print("Filas:", len(df))
print("AROs diferentes:", df["aro"].nunique())


# ============================================================
# 5. ARO como string
# ============================================================

df["aro"] = df["aro"].astype(str)


# ============================================================
# 6. Añadir información CARD
# ============================================================

def get_card_value(aro, field):

    return aro_index.get(
        str(aro),
        {}
    ).get(field)


df["aro_name"] = df["aro"].apply(
    lambda x: get_card_value(x, "aro_name")
)

df["aro_description"] = df["aro"].apply(
    lambda x: get_card_value(x, "aro_description")
)

df["amr_gene_family"] = df["aro"].apply(
    lambda x: get_card_value(x, "amr_gene_family")
)

df["drug_class"] = df["aro"].apply(
    lambda x: get_card_value(x, "drug_class")
)

df["resistance_mechanism"] = df["aro"].apply(
    lambda x: get_card_value(
        x,
        "resistance_mechanism"
    )
)

df["card_model_id"] = df["aro"].apply(
    lambda x: get_card_value(
        x,
        "card_model_id"
    )
)

df["model_type"] = df["aro"].apply(
    lambda x: get_card_value(
        x,
        "model_type"
    )
)


# ============================================================
# 7. Normalizar nombres de antibióticos
# ============================================================

def antibiotic_class(antibiotic):

    antibiotic = str(antibiotic).lower().strip()

    # --------------------------------------------------------
    # Fluoroquinolonas
    # --------------------------------------------------------

    fluoroquinolones = {
        "ciprofloxacin",
        "levofloxacin",
        "ofloxacin",
        "norfloxacin",
        "moxifloxacin",
        "enrofloxacin",
        "marbofloxacin",
        "danofloxacin",
        "difloxacin",
    }

    if antibiotic in fluoroquinolones:
        return "fluoroquinolone"

    # --------------------------------------------------------
    # Aminoglucósidos
    # --------------------------------------------------------

    aminoglycosides = {
        "gentamicin",
        "amikacin",
        "tobramycin",
        "streptomycin",
        "kanamycin",
        "neomycin",
    }

    if antibiotic in aminoglycosides:
        return "aminoglycoside"

    # --------------------------------------------------------
    # Tetraciclinas
    # --------------------------------------------------------

    tetracyclines = {
        "tetracycline",
        "doxycycline",
        "minocycline",
        "oxytetracycline",
    }

    if antibiotic in tetracyclines:
        return "tetracycline"

    # --------------------------------------------------------
    # Penicilinas
    # --------------------------------------------------------

    penicillins = {
        "ampicillin",
        "amoxicillin",
        "penicillin",
        "piperacillin",
        "oxacillin",
    }

    if antibiotic in penicillins:
        return "penicillin"

    # --------------------------------------------------------
    # Cefalosporinas
    # --------------------------------------------------------

    cephalosporins = {
        "ceftriaxone",
        "cefotaxime",
        "ceftazidime",
        "cefepime",
        "cefoxitin",
        "cephalothin",
        "cephalexin",
    }

    if antibiotic in cephalosporins:
        return "cephalosporin"

    # --------------------------------------------------------
    # Macrólidos
    # --------------------------------------------------------

    macrolides = {
        "erythromycin",
        "azithromycin",
        "clarithromycin",
    }

    if antibiotic in macrolides:
        return "macrolide"

    # --------------------------------------------------------
    # Fenicoles
    # --------------------------------------------------------

    phenicols = {
        "chloramphenicol",
        "florfenicol",
    }

    if antibiotic in phenicols:
        return "phenicol"

    # --------------------------------------------------------
    # Sulfonamidas
    # --------------------------------------------------------

    sulfonamides = {
        "sulfamethoxazole",
        "sulfisoxazole",
        "sulfadiazine",
    }

    if antibiotic in sulfonamides:
        return "sulfonamide"

    # --------------------------------------------------------
    # Otros
    # --------------------------------------------------------

    return "unknown"


df["antibiotic_class"] = df["antibiotic"].apply(
    antibiotic_class
)


# ============================================================
# 8. Determinar si el ARO es relevante para el antibiótico
# ============================================================

def is_relevant(row):

    drug_class = str(
        row["drug_class"]
    ).lower()

    antibiotic_class_name = str(
        row["antibiotic_class"]
    ).lower()

    if (
        not drug_class
        or drug_class == "nan"
        or antibiotic_class_name == "unknown"
    ):
        return False

    # --------------------------------------------------------
    # Comparamos la clase del antibiótico con drug_class CARD
    # --------------------------------------------------------

    return antibiotic_class_name in drug_class


df["relevant_to_antibiotic"] = df.apply(
    is_relevant,
    axis=1
)


# ============================================================
# 9. AROs no encontrados
# ============================================================

no_encontrados = df[
    df["aro_name"].isna()
]["aro"].unique()

print(
    "\nAROs no encontrados en CARD:",
    len(no_encontrados)
)

if len(no_encontrados) > 0:
    print(no_encontrados)


# ============================================================
# 10. Guardar tabla completa anotada
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    "\nArchivo completo guardado:"
)

print(OUTPUT_FILE)


# ============================================================
# 11. Mostrar resultados relevantes
# ============================================================

relevantes = df[
    df["relevant_to_antibiotic"]
].copy()

print("\n========================================")
print("CANDIDATOS RELEVANTES")
print("========================================")

print(
    relevantes[
        [
            "accession",
            "antibiotic",
            "phenotype",
            "aro",
            "aro_name",
            "drug_class",
            "resistance_mechanism",
        ]
    ].to_string(index=False)
)


# ============================================================
# 12. Guardar solamente candidatos
# ============================================================

relevantes.to_csv(
    CANDIDATES_FILE,
    index=False
)

print(
    "\nArchivo de candidatos guardado:"
)

print(CANDIDATES_FILE)

print(
    "\nNúmero de filas candidatas:",
    len(relevantes)
)

print(
    "Número de genomas con al menos "
    "un candidato:",
    relevantes["accession"].nunique()
)