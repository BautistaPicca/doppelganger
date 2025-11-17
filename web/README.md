# Tesina Proyecto 2025 – Frontend

Aplicación web que interactúa con la API de reconocimiento facial para registro, login y búsqueda de rostros similares.

---

## Características principales

### Rutas y funcionalidades

- **`/login` / `/register`**
  - Permite al usuario registrarse o iniciar sesión utilizando **reconocimiento facial** y datos tradicionales en el caso del registro (nombre, email, contraseña).  
  - Las imágenes se envían al backend para generar embeddings y validar coincidencias.  
  - La sesión se mantiene mediante **JWT**.

- **`/match`**
  - Permite subir una imagen y recibir los K rostros más parecidos del sistema.  
  - Se muestra nombre, profesión y nivel de confianza de la coincidencia.  

- **Datos de famosos**
  - Las fotos y descripciones de celebridades se obtienen automáticamente desde **Wikipedia**.  
  - Se incluye una breve descripción de la profesión de cada famoso.

---

## Tecnologías y librerías

- **UI**
  - [ShadCN/UI](https://ui.shadcn.com/) para componentes de interfaz modernos y consistentes.  
  - [Framer Motion](https://www.framer.com/motion/) para animaciones responsivas en los componentes.

- **Autenticación**
  - Uso de **JWT** para mantener la sesión de usuario y asegurar la comunicación con la API.  

---

## Flujo de usuario

1. **Registro/Login**
   - Usuario se registra o inicia sesión enviando datos y/o imagen facial.  
   - El backend valida la imagen y devuelve un token JWT si la autenticación es exitosa.

2. **Búsqueda de parecidos**
   - Usuario sube una foto en la ruta `/match`.  
   - La app envía la imagen al backend y recibe los K parecidos más relevantes.  
   - Se muestran resultados con foto, nombre y descripción de profesión.

3. **Gestión de sesión**
   - El JWT se almacena localmente y se utiliza para mantener la sesión activa.  
   - Los usuarios pueden cerrar sesión de forma segura, eliminando el token.

---