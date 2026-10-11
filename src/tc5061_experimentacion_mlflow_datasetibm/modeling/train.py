
# Entrenamiento del modelo y seguimiento con MLflow

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay,)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tc5061_experimentacion_mlflow_datasetibm.features import (get_project_root,load_data,prepare_data,)


def obtener_argumentos():
    "Define los parametros"
    parser = argparse.ArgumentParser(description="Exp con MLflow")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", dest="max_iter", type=int, default=1000)
    parser.add_argument("--class-weight", dest="class_weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", dest="test_size", type=float, default=0.2)
    return parser.parse_args()


def crear_modelo(var_num, var_cat, args):
    "Prepara las columnas y crea el modelo de regresión logística"

    # Las col num se escalan.
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())])

    # Las col cat son indicadores.
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocesamiento = ColumnTransformer([
        ("numericas", num_pipeline, var_num),
        ("categoricas", cat_pipeline, var_cat)])

    peso = None if args.class_weight == "none" else "balanced"

    modelo = Pipeline([
        ("preprocesamiento", preprocesamiento),
        ("clasificador", LogisticRegression(
            C=args.C, max_iter=args.max_iter, class_weight=peso, 
            random_state=args.seed))])

    return modelo


def main():
    args = obtener_argumentos()

    if args.C <= 0 or args.max_iter <= 0:
        raise ValueError("C y max-iter mayor a 0")

    if not 0 < args.test_size < 1:
        raise ValueError("test-size debe estar entre 0 y 1.")

    # Cargar y preparar los datos
    df = load_data()
    X, y, var_num, var_cat = prepare_data(df)

    # Semilla
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size,
        random_state=args.seed, stratify=y)

    modelo = crear_modelo(var_num, var_cat, args)

    # Resultados MLFLOW
    raiz = get_project_root()
    reports = raiz / "reports"
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    db_path = (raiz / "mlflow.db").resolve()
    #mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("IBM_Telco_Customer_Churn_Experimentos")

    with mlflow.start_run():
        modelo.fit(X_train, y_train)

        predicciones = modelo.predict(X_test)
        probabilidades = modelo.predict_proba(X_test)[:, 1]

        # Metricas de resultadps
        metricas = {
            "accuracy": accuracy_score(y_test, predicciones),
            "precision": precision_score(y_test, predicciones, zero_division=0),
            "recall": recall_score(y_test, predicciones, zero_division=0),
            "f1": f1_score(y_test, predicciones, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilidades)}

        # Guardamos
        mlflow.log_params({
            "C": args.C, "max_iter": args.max_iter,
            "class_weight": args.class_weight,
            "seed": args.seed, "test_size": args.test_size})
        mlflow.log_metrics(metricas)

        # Guardamos como archivo
        reports.mkdir(parents=True, exist_ok=True)
        metricas_path = reports / "metrics.json"
        with open(metricas_path, "w", encoding="utf-8") as archivo:
            json.dump(metricas, archivo, indent=4)

        mlflow.log_artifact(str(metricas_path), artifact_path="reports")

        # Generamos la matriz de confusión
        grafica = figures / "confusion_matrix.png"
        ConfusionMatrixDisplay.from_predictions(
            y_test, predicciones,
            display_labels=["No churn", "Churn"], values_format="d")
        plt.tight_layout()
        plt.savefig(grafica, dpi=150)
        plt.close()

        mlflow.log_artifact(str(grafica), artifact_path="figures")

        # Registramos el pipeline completo
        mlflow.sklearn.log_model(modelo, artifact_path="model")

        print("\nMétricas obtenidas:")
        for nombre, valor in metricas.items():
            print(f"{nombre}: {valor:.4f}")

        print(f"\nRun registrado: {mlflow.active_run().info.run_id}")
        print(f"Base de datos: {db_path}")


if __name__ == "__main__":
    main()