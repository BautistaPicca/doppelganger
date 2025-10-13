embedder/main.py es el programa que se encarga de procesar un dataset de "celebridades" y almacenar sus embeddings, metadatos, etc.
Esto para la aplicación recreativa del proyecto, el acceso por reconocimiento facial no requeriría esto, se irían añadiendo de a poco durante
la ejecución del programa en vez de preprocesar las cosas.

indexer/.main.py es el programa que se encarga de tomar un dataset procesado y crear la base de datos vectorial, también esto sólo sirve para la aplicación recreativa, ya que esto tambien se trata de un preprocesamiento por nuestra parte.

En el acceso por reconocimiento facial se debería extraer el / los embeddings de la imagen y en ese mismo instante añadirlo a la base de datos vectorial (que a diferencia de la aplicación recreativa no se borraría cada vez que modificamos el dataset)