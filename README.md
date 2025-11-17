<<<<<<< Updated upstream
# face-matcher

Descripción de la maqueta que se encuentra en el repositorio: Maqueta base que hicimos como primer paso, utilizando Google Colab, con framework TensorFlow y modelo preentrenado MobileNetV2. Al querer obtener más mejoras, como aumentar los entrenamientos y por una cuestión de velocidad al entrenar,  decidimos migrar al framework Pytorch para que pueda utilizar la GPU que tenemos y entrenar más rápido. Esto se encuentra dentro de ai_engine.
=======
# Tesina Proyecto 2025 – UNRC (Río Cuarto)
Este proyecto forma parte de la tesina 2025 de la materia "Proyecto" para la Universidad Nacional de Río Cuarto.  
La idea principal es desarrollar una aplicación recreativa donde el usuario pueda subir una foto y ver **a qué celebridad se parece**.

Para lograrlo se entrenó un **clasificador basado en una red neuronal convolucional**, que toma una imagen de rostro y predice (en base a probabilidades) a qué clase/persona pertenece.

---

## 🧠 Descripción general

El sistema está compuesto por dos partes:

- **Entrenamiento del modelo**: donde se construye el clasificador y se entrena con un dataset de rostros.
- **Inferencia**: donde se carga el modelo entrenado y se usa para procesar fotos de usuarios.

En la etapa de inferencia, el modelo devuelve por probabilidad a qué clase pertenece la imagen ingresada, por cada persona del dataset hay una clase.

---

## 📦 Entrenamiento del clasificador

Para iniciar el entrenamiento ejecutar:

```bash
python -m ai_engine.model.classifier.train
```

Cuando corras el script, te pedirá lo siguiente:

- Seleccionar carpeta del dataset

Se abrirá un explorador de archivos para que elijas la carpeta del dataset.

Si no tenés un dataset, o si cerrás la ventana / tocás Cancelar, el sistema descargará automáticamente uno desde Google Drive.

- Seleccionar carpeta de salida (output)

En esta carpeta se guardarán dos archivos:
1. Un archivo .pth con los pesos del modelo entrenado.
2. Un archivo .json que mapea cada índice de clase con una persona.

Ambos archivos son necesarios para poder hacer inferencia.

## 🚀 Inferencia (uso del modelo entrenado)
Para usar el modelo entrenado, el proyecto incluye el servicio:

**ClassifierService**

Este servicio carga automáticamente:
 1. El modelo .pth
 2. El archivo .json de mapeo

desde la carpeta:
```bash
run/models
```
de la carpeta raíz del proyecto.

## 📌 Notas finales

- El proyecto es de carácter recreativo y experimental.
- Sirve como base para entender cómo integrar una red convolucional en una aplicación real.

## 🛠️ Scripts

Durante el desarrollo del proyecto creamos varios **scripts utilitarios** para procesar imágenes, preparar datasets, generar gráficas y realizar distintas tareas auxiliares.

Todos estos scripts se encuentran en:

```bash
/ai_engine/tasks/
```

## 🔍 Funcionalidad secundaria: Reconocimiento facial

Además de la parte recreativa del proyecto ("a qué celebridad te parecés"), también exploramos una funcionalidad más técnica: permitir iniciar sesión en un sistema usando reconocimiento facial.

La idea era entrenar una CNN vectorizadora, es decir, una red cuyo objetivo fuera extraer características faciales y devolver un embedding (un vector de características).
Con esos vectores planeábamos:

- Guardarlos en una base de datos
- Realizar búsquedas por similitud usando distancia coseno
- Determinar si dos imágenes pertenecen a la misma persona

❗ Dificultades encontradas

Entrenar desde cero un modelo de este tipo resultó ser un desafío GIGANTE.
Los modelos comerciales que se usan hoy en día (ArcFace, FaceNet, etc)
- Son gigantes en cantidad de parámetros
- Están entrenados con millones de imágenes
- Requieren hardware muy potente y muchísimo tiempo de entrenamiento

A pesar de varios intentos, pruebas, limpieza de datasets, técnicas de data augmentation y muchas horas de cómputo, los resultados obtenidos fueron prácticamente nulos para un uso real.

Evaluamos usar un backbone pre-entrenado y entrenar solo las últimas capas con un dataset propio.
Sin embargo, esto solo es útil en escenarios muy específicos, por ejemplo:
* Registrar acceso de empleados en horario nocturno, al aire libre, bajo condiciones controladas.
En ese caso sí es útil entrenar un modelo adaptado al contexto de iluminación y cámara.

📦 Estado actual de esta funcionalidad
Decidimos conservar todo el código relacionado al vectorizador dentro del proyecto.
Incluye:
- Ejemplo de cómo debería funcionar usando un modelo pre-entrenado real
- Opción para seleccionar el último modelo que entrenamos nosotros
- Código estructurado para permitir inferencia, embedding y búsquedas

Sin embargo, es importante aclarar que el modelo entrenado por nuestra cuenta no ofrece resultados válidos.
Se mantiene solo para documentar el proceso.
>>>>>>> Stashed changes
