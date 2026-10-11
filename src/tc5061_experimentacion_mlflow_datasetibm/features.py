
# Funciones para cargar y preparar los datos

from pathlib import Path
import pandas as pd


KAGGLE_DATASET = "blastchar/telco-customer-churn"
TARGET_COLUMN = "Churn"


def get_project_root() -> Path:
    """Devuelve la carpeta principal del proyecto."""
    return Path(__file__).resolve().parents[2]


def find_csv(directory: Path) -> Path:
    """Busca un archivo CSV dentro de una carpeta."""
    files = list(directory.rglob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No se encontró un CSV en {directory}")

    # Si hay varios archivos, preferimos el que parezca ser el de Telco.
    for file in files:
        if "customer" in file.name.lower() or "telco" in file.name.lower():
            return file

    return files[0]


def load_data() -> pd.DataFrame:
    """Carga los datos locales o los descarga desde Kaggle."""
    project_root = get_project_root()
    raw_directory = project_root / "data" / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)

    if list(raw_directory.rglob("*.csv")):
        csv_path = find_csv(raw_directory)
        print(f"Leyendo archivo: {csv_path}")
    else:
        import kagglehub

        downloaded_directory = Path(
            kagglehub.dataset_download(KAGGLE_DATASET)
        )
        csv_path = find_csv(downloaded_directory)
        print(f"Archivo descargado: {csv_path}")

    return pd.read_csv(csv_path)


def prepare_data(df: pd.DataFrame):
    """Limpia los datos y separa las variables predictoras del objetivo."""
    data = df.copy()
    data.columns = data.columns.str.strip()

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"No se encontró la columna {TARGET_COLUMN}")

    # TotalCharges puede tener espacios en lugar de valores numéricos.
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(
            data["TotalCharges"].astype(str).str.strip().replace("", pd.NA),
            errors="coerce"
        )

    # Quitamos duplicados exactos y registros sin objetivo.
    data = data.drop_duplicates()
    data = data.dropna(subset=[TARGET_COLUMN])

    # Convertimos la variable que queremos predecir a 0 y 1.
    valores = data[TARGET_COLUMN].astype(str).str.strip().str.lower()
    y = valores.map({"no": 0, "yes": 1, "0": 0, "1": 1})

    valid_rows = y.notna()
    data = data.loc[valid_rows].copy()
    y = y.loc[valid_rows].astype(int)

    # El identificador no se utiliza como variable predictora.
    columnas_excluir = [TARGET_COLUMN]
    if "customerID" in data.columns:
        columnas_excluir.append("customerID")

    X = data.drop(columns=columnas_excluir)

    # Separamos las columnas por su tipo actual.
    var_num = X.select_dtypes(include=["number"]).columns.tolist()
    var_cat = X.select_dtypes(exclude=["number"]).columns.tolist()

    print(f"Registros: {len(X)}")
    print(f"Variables predictoras: {X.shape[1]}")
    print(f"Variables numéricas: {len(var_num)}")
    print(f"Variables categóricas: {len(var_cat)}")

    return X, y, var_num, var_cat
