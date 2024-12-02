# Proyecto de detección de Parkinson

## Índice
* [Descripción](#descripción)
* [Habilidades](#habilidades)
* [Cómo clonar el repositorio y ejecutar el código](#cómo-clonar-el-repositorio-y-ejecutar-el-código)
* [Posibles problemas](#posibles-problemas)

## Descripción
Este es un proyecto de ciencia de datos para predecir si un sujeto tiene o no Parkinson a partir de ciertas características de la voz. La enfermedad de Parkinson afecta a la voz de diferentes maneras y se han recogido datos sobre las particularidades de la voz de sujetos sanos y sujetos con Parkinson. El objetivo de este proyecto es poder predecir a tiempo la enfermedad en una etapa temprana para poder actuar cuanto antes.

Enlace al dataset: [Kaggle]("https://www.kaggle.com/datasets/jainaru/parkinson-disease-detection")

## Habilidades

En este proyecto se pueden aprecier muchas habilidades de data science como:

* Uso de librerías externas
* Visualización de datos
* Tratameinto y transformación de datos
* Generación de datos sintéticos
* Análisis de datos
* Automatización de tareas
* Uso de algoritmos de Machine Learning y Deep Learning

## Cómo clonar el repositorio y ejecutar el código
### Clonación
`git clone https://github.com/Toni2Morales/ProyectoNLP.git`

### Previo a la ejecución
**Necesitas tener una versión de python inferior a la 3.11.0**

**Necesitas comprobar si tienes instalado Graphviz:**

Tienes varias maneras de tener el entorno idoneo para ejecutar el código del repositorio:

* Creando tu propio entorno virtual
    1. Abrir una terminal y crearlo: ```virtualenv -p python3.10 env```
    2. Activarlo
        - Unix/MacOS: `. env/bin/activate`
        - Windows: `env\Scripts\activate`
    3. Instalar los requerimientos: 
        - `pip install -r requerimientos.txt`
        - `python -m pip install -r requerimientos.txt`
    4. Ejecutar el código
    5. Desactivarlo: `deactivate`
## Posibles problemas
* `failed to execute PosixPath('dot')`: Instalar Graphviz
    * MacOS: Ejecutar `sudo port install graphviz` en un terminal.
    * Linux: Ejecutar `sudo apt install graphviz` en un terminal.
    * Windows(Chocolatey instalado): Ejecutar `choco install graphviz` en un terminal.
* `font family ['serif'] not found.`: Ejecutar la sección `"Previo"` de nuevo, ejecutar el código de la gráfica otra vez y continuar.
* `libGL.so.1: cannot open shared object file: No such file or directory`: Instalar todas las siguientes dependencias:
    * sudo apt-get install libgl1-mesa-glx
    * sudo apt-get install libglib2.0-0
    * sudo apt-get install ffmpeg libsm6 libxext6
