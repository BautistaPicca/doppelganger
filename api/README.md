# Tesina Proyecto 2025 – UNRC (Río Cuarto)

API de la aplicación para la comunicación entre cliente y servidor.  
La autenticación mediante rostro se implementa usando **JSON Web Tokens (JWT)**.

---
## Como usar
Pasos previos:
 - Activar el ambiente venv
 ```bash
  .\venv\Scripts\activate
 ```
 - Instalar dependencias
 ```bash
  pip install -r requeriments.txt
 ```
Para iniciar el servidor:
```bash
  python -m api.app
```
## Rutas disponibles

- **`/api/auth/login`**  
  Inicia sesión enviando una imagen del rostro.  
  - Respuesta exitosa: JWT generado y enviado al usuario.

- **`/api/auth/register`**  
  Registra un usuario con los siguientes datos:  
  - Nombre completo  
  - Nombre de usuario  
  - Correo electrónico  
  - Contraseña  
  - Imagen del rostro

- **`/api/auth/profile`**  
  Devuelve información del usuario, **excluyendo la contraseña**.

- **`/api/match`**  
  Envía una imagen y recibe los K rostros más parecidos:  
  - Nombre del usuario  
  - Confianza de la coincidencia

- **`/api/config`**  
  Configura el modelo utilizado para el vectorizador facial.

---

## Auth Service

El **Auth Service** gestiona la autenticación por reconocimiento facial y ofrece:

- **Búsqueda eficiente**: utiliza FAISS para búsquedas rápidas (O(log n)).  
- **Detección de duplicados**: evita registrar el mismo rostro dos veces.  
- **Validaciones de seguridad**: control de matches ambiguos y verificación de contraseñas.

### Funcionalidades principales

1. **Registro de usuario**
   - Guarda datos básicos (`username`, `name`, `email`) y la imagen del rostro.  
   - Genera un embedding vectorial del rostro.  
   - Verifica duplicados antes de registrar un nuevo usuario.  
   - Almacena el embedding en un índice FAISS.

2. **Autenticación por rostro**
   - Genera un embedding de la imagen recibida.  
   - Busca en el índice el usuario más similar.  
   - Evalúa coincidencias según un umbral configurable y detecta matches ambiguos.  
   - Retorna el usuario autenticado y la confianza de la coincidencia.

3. **Configuración**
   - Permite cambiar la configuración del vectorizador y el índice dinámicamente.

---

### Clase `AuthUser`

Representa a cada usuario registrado:

- `username`, `name`, `email`, `password_hash`  
- `face_image_path` : ruta local de la imagen del rostro  
- `registered_at`: fecha de registro  

Incluye métodos para:

- Verificar contraseñas (`check_password`)  
- Serializar a diccionario para APIs (`to_dict`)  

---

Con esta estructura, la API y el servicio de autenticación están organizados para permitir un flujo seguro y eficiente de registro y login por rostro.
