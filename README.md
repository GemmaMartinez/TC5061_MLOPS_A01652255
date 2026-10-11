# Actividad Individual | Código de experimentación base con MLFlow
### Gemma Alejandra Martínez Santos
### A01652255

## 1. Descripción

Este proyecto funciona para entrenar un modelo de Machine Learning, teniendo como base el notebook trabajado en la actividad anterior mediante Jupyter Notebook en Anaconda. Se utiliza el conjunto de datos IBM Telco Customer Churn para predecir si un cliente abandona el servicio.

Para esta actividad, el entrenamiento se llevó a un script de Python dentro de un entorno virtual, buscando que los experimentos sean más ordenados.

Ahora con lo visto en clase quise probar con visual code por la facilidad de las carpetas e ir viendo donde trabajas, se me hizo un poco más dinámico y ordenado. Trabajé desde la terminal de powershell y la de Visual Studio; configure la carpeta en donde tenían que estar, se utilizó py 3.14, cookie cutter data science, también se crearon docs manualmente.

## 2. Objetivo

Se busca que se entrenen los datos y se configuren los hiperparametros, así como el train y test; para posteriormente registrar parámetros, métricas, modelo y artefactos en MLflow y poderlo visualizar de forma más dinámica.

## 3. Estructura del proyecto

Como he mencionado la carpeta se creo con cookiecutter y algunas carpetas manualmente, se implemento esta forma por lo que hemos visto en clase, para poder ir familirizandome con esta herramienta

TC5061_Experimentacion_MLFlow_DataSetIBM/    
├── data/raw    
├── mlruns   
├── notebooks/    
│   └── TC5061_Notebook_Baseline_MartinezGemma.ipynb     
├── reports/   
│   └── figures/   
│   └── metrics.json/   
├── src/tc5061_experimentacion_mlflow_datasetibm/   
│   └── modeling/   
│     └── _int_.py   
│     └── predict.py    
│     └── train.py    
│     └── _int_.py   
│     └── features.py    
├── README.md   
└── requirements.txt    

### features.py

Se encarga de localizar y cargar el conjunto de datos. Si no encuentra un CSV en `data/raw`, intenta descargarlo mediante KaggleHub. También limpia los nombres de las columnas, preparando las variables

### train.py

Contiene el entrenamiento. Divide los datos en conjuntos de entrenamiento y prueba, construye el pipeline de preprocesamiento y entrena una regresión logística. Calcula métricas de evaluación y registra los resultados con MLflow.

### notebooks/

Contiene el notebook de la tara previa y aquí se ponen todos los que vamos ocupando.

## 4. Modelo y preprocesamiento

Se utiliza regresión logística para realizar una clasificación binaria.

Las variables numéricas se procesan mediante estandarización. Las variables categóricas con imputación (One-Hot y Simple Imputer).

La variable `Churn` se transforma en una etiqueta binaria: 0 para NO y 1 para YES.

REPRODUCIBILIDAD 

Para mantener la reproducibilidad de los experimentos se otorgó una semilla de 42, es utilizada tanto en la división de los datos de train y test como en el modelo de regresión logística.
La semilla también se encuentra parametrizada mediante --seed, por lo que puede modificarse desde la línea de comandos. El experimento mantiene los mismos datos, es decir las mismas condiciones.

Las dependencias de lo que fue utilizado, así como sus versiones y un poco más de detalle, se encuentran especificadas en el archivo requirements.txt.

## 5. Parámetros

A continuación mostramos los párametros y lo que significa, se intentó poner como comentario de igual forma dentro del code

- `--C`: inverso de la fuerza de regularización
- `--max-iter`: máximo de iteraciones
- `--class-weight`: clases `none` y `balanced`
- `--seed`: semilla aleatoria
- `--test-size`: proporción de datos

## 6. Métricas y artefactos

Se registran las siguientes métricas:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- ROC-AUC.

Los experimentos se almacenan en el experimento de MLflow.

## 7. Instalación y ejecución

Desde PowerShell, en la raíz del proyecto, se puede activar el entorno virtual y establecer la ruta de los módulos:

se realizó desde powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$PWD\src"

## 8. Experimentos

Se comparan diferentes configuraciones variando `C`, `max-iter` y `class-weight`. 

PS C:\Users\gemma\Documents\MLOPS\WK4\TC5061_Experimentacion_MLFlow_DataSetIBM> python -m tc5061_experimentacion_mlflow_datasetibm.modeling.train --C 1.0 --max-iter 1000 --class-weight none --seed 42

PS C:\Users\gemma\Documents\MLOPS\WK4\TC5061_Experimentacion_MLFlow_DataSetIBM> python -m tc5061_experimentacion_mlflow_datasetibm.modeling.train --C 0.1 --max-iter 1000 --class-weight none --seed 42

PS C:\Users\gemma\Documents\MLOPS\WK4\TC5061_Experimentacion_MLFlow_DataSetIBM> python -m tc5061_experimentacion_mlflow_datasetibm.modeling.train --C 10.0 --max-iter 1000 --class-weight none --seed 42

PS C:\Users\gemma\Documents\MLOPS\WK4\TC5061_Experimentacion_MLFlow_DataSetIBM> python -m tc5061_experimentacion_mlflow_datasetibm.modeling.train --C 1.0 --max-iter 2000 --class-weight none --seed 42

PS C:\Users\gemma\Documents\MLOPS\WK4\TC5061_Experimentacion_MLFlow_DataSetIBM> python -m tc5061_experimentacion_mlflow_datasetibm.modeling.train --C 1.0 --max-iter 1000 --class-weight balanced --seed 42

## 9. Conclusión
Mediante este trabajo, además de realizar los experimentos, francamente me ayudó mucho el poder ordenar mis capetas, como no estoy tan acostumbrada a hacerlo, me costo un poco hacerlo. También me gustó interactuar con MLFlow porque mi versión se actualizó y pude trabajar con esta nueva interfaz. 

Nota: Para la parte de análisis de resultados y capturas, se anexan al pdf.

## 10. Referencias
ChatGpt. (s. f.). ChatGpt. https://chatgpt.com/
Se utilizó para verificar que cumpla el trabajo con todos los requisitos, errores ortograficos y errores de code    
Gemini. (s. f.). gemini. https://gemini.google.com/
Se utilizó para verificar que cumpla el trabajo con todos los requisitos, errores ortograficos y errores de code    
Python 3.14 documentation. (s. f.). Python Documentation. https://docs.python.org/3.14/    
MLflow — Open Source AI Platform for Agents, LLMs & Models. (s. f.). MLflow. https://mlflow.org/docs/latest/    

