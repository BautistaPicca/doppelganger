from datetime import datetime, timedelta, timezone
import io
from flask import Blueprint, request, jsonify, current_app
import jwt
from functools import wraps
from PIL import Image

from ai_engine.utils.pre_processing import pad_and_resize

auth_bp = Blueprint("api/auth", __name__)
auth_service = None

@auth_bp.record_once
def on_load(state):
    """Inicializa el servicio de autenticación al cargar el blueprint"""
    global auth_service
    from api.services.auth_service import FaceAuthService
    
    auth_service = FaceAuthService(
        upload_dir=state.app.config["UPLOAD_FOLDER"],
        index_dir=state.app.config.get("FACE_INDEX_DIR", "run/server/face_index"),
        threshold=state.app.config.get("FACE_THRESHOLD", 0.65)
    )
    
    print(f"FaceAuthService inicializado: {auth_service.get_stats()}")

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'message': 'Token faltante'}), 401
        
        try:
            if token.startswith('Bearer '):
                token = token[7:]
            
            data = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = data['username']
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token expirado'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Token inválido'}), 401
        
        return f(current_user, *args, **kwargs)
    
    return decorated

def generate_token(username, name):
    token = jwt.encode({
        'username': username,
        'name': name,
        'exp': datetime.now(timezone.utc) + timedelta(hours=24)
    }, current_app.config['SECRET_KEY'], algorithm='HS256')
    return token

@auth_bp.route('/login', methods=['POST'])
def login():
    """
    Login con reconocimiento facial.
    """
    if 'image' not in request.files:
        return jsonify({'message': 'Imagen no proporcionada'}), 400
    
    image_file = request.files['image']
    
    if image_file.filename == '':
        return jsonify({'message': 'Archivo vacío'}), 400
    
    try:
        # Leer imagen
        file_bytes = image_file.read()
        img = Image.open(io.BytesIO(file_bytes))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        
        user, message, confidence = auth_service.authenticate(
            face_image=img
        )
        
        if user is None:
            # Login fallido
            return jsonify({
                'message': message,
                'confidence': confidence
            }), 401
        
        # Login exitoso
        token = generate_token(user.username, user.name)
        
        return jsonify({
            'token': token,
            'user': user.to_dict(),
            'confidence': confidence,
            'message': message
        }), 200
        
    except ValueError as e:
        return jsonify({'message': str(e)}), 400
    except Exception as e:
        return jsonify({'message': f'Error al procesar imagen: {str(e)}'}), 500
    
@auth_bp.route('/register', methods=['POST'])
def register():
    """
    Registra un nuevo usuario con contraseña y rostro.
    La contraseña es meramente para asemejarse a un sistema real, no se usa para autenticación.

    Form:
        - username: str (requerido)
        - password: str (requerido)
        - name: str (requerido)
        - email: str (requerido)
        - image: file (requerido)
    """
    try:
        # Obtener datos del formulario
        username = request.form.get('username')
        password = request.form.get('password')
        name = request.form.get('name')
        email = request.form.get('email')
        
        # Validaciones básicas
        if not all([username, password, name, email]):
            return jsonify({'message': 'Faltan campos requeridos: username, password, name, email'}), 400
        
        image_file = request.files['image']
        if image_file.filename != '':
            # Leer y convertir el archivo a Image
            file_bytes = image_file.read()
            face_image = Image.open(io.BytesIO(file_bytes))
            
            if face_image.mode != 'RGB':
                face_image = face_image.convert('RGB')
            
            # Ajustar
            face_image = pad_and_resize(face_image)
        
        # Crear usuario
        user, image_path = auth_service.create_user(
            username=username,
            password=password,
            name=name,
            email=email,
            face_image=face_image
        )
        
        return jsonify({
            'message': 'Usuario registrado exitosamente',
            'user': user.to_dict(),
            'has_face_auth': image_path is not None
        }), 201
        
    except ValueError as e:
        return jsonify({'message': str(e)}), 400
    except Exception as e:
        return jsonify({'message': f'Error al registrar: {str(e)}'}), 500

@auth_bp.route('/profile', methods=['GET'])
@token_required
def profile(current_username):
    """Obtiene el perfil completo del usuario actual"""
    user = auth_service.get_user(current_username)
    
    if user is None:
        return jsonify({'message': 'Usuario no encontrado'}), 404
    
    return jsonify({
        'user': user.to_dict()
    }), 200