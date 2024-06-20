# Proyecto de detección de Parkinson
-------------
## Descripción
Este es un proyecto de ciencia de datos para predecir si un sujeto tiene o no Parkinson a partir de ciertaqs características de la voz. La enfermedad de Parkinson afecta a la voz de diferentes maneras y se han recogido datos sobre las particularidades de la voz de sujetos sanos y sujetos con Parkinson. El objetivo de este proyecto es poder predecir a tiempo la enfermedad en una etapa temprana para poder actuar cuanto antes.

Enlace al dataset: [Kaggle]("https://www.kaggle.com/datasets/jainaru/parkinson-disease-detection")

## Cómo clonar el repositorio y ejecutar el código
### Clonación
`git clone https://github.com/Toni2Morales/ProyectoNLP.git`

### Previo a la ejecución
**Necesitas tener una versión de python inferior a la 3.11.0**

Tienes varias maneras de tener el entorno idoneo para ejecutar el código del repositorio:

* Ejecutándo el código indicando el entorno virtual de la carpeta `env/bin/python3.10` que viene en el repositorio(Solo en Linux).
* Creando tu propio entorno virtual
    1. Crearlo: `virtualenv -p python3.10 env`
    2. Activarlo
        - Unix/MacOS: `. env/bin/activate`
        - Windows: `env\Scripts\activate`
    3. Instalar los requerimientos: 
        - pip install -r requerimientos.txt
        - python -m pip install -r requerimientos.txt
    4. Ejecutar el código
    5. Desactivarlo: `deactivate`
* A través de Docker
----instrucciones a escribir--------------------------
## Posibles problemas
* `failed to execute PosixPath('dot')`: Instalar Graphviz
    * MacOS: `sudo port install graphviz`
    * Linux: `sudo apt install graphviz`
    * Windows(Chocolatey instalado): `choco install graphviz`
* `font family ['serif'] not found.`: Ejecutar la sección `"Previo"` de nuevo, ejecutar el código de la gráfica otra vez y continuar.