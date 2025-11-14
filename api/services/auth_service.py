from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
from PIL import Image
from torchvision import transforms
from werkzeug.security import generate_password_hash, check_password_hash

from api.config import CONFIGS

@dataclass
class AuthUser:
    """
    Modelo de usuario para autenticación con reconocimiento facial.
    Se serializa automáticamente por CompositeIndexerService.
    """
    username: str
    name: str
    email: str
    password_hash: str
    face_image_path: Optional[str] = None
    registered_at: Optional[str] = None
    
    def check_password(self, password: str) -> bool:
        """Verifica si la contraseña es correcta"""
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self) -> dict:
        """Convierte el usuario a diccionario para JSON"""
        return {
            'username': self.username,
            'name': self.name,
            'email': self.email,
            'has_face_auth': self.face_image_path is not None,
            'registered_at': self.registered_at
        }
    
class FaceAuthService:
    """
    Servicio de autenticación con reconocimiento facial.
    
    Características:
    - Persistencia en disco
    - Búsqueda eficiente O(log n) con FAISS para gran cantidad de usuarios
    - Detección de duplicados al registrar nuevos usuarios
    - Validaciones básicas de seguridad
    """
    def __init__(self, upload_dir, index_dir: str = "run/server/face_index", config_name="pretrained"):
        """
        Inicializa el servicio de autenticación facial.
        
        Args:
            upload_dir: Directorio para guardar imágenes de rostros
            index_dir: Directorio para guardar el índice FAISS
        """

        self.upload_dir = upload_dir
        self.index_dir = Path(index_dir)

        # Crea los directorios si no existen
        os.makedirs(self.upload_dir, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.apply_config(config_name)

    def apply_config(self, config_name: str):
        """
        Cambia dinámicamente el vectorizador y el índice según la configuración seleccionada.
        
        Cuando se aplica una nueva configuración, se reinicia la base de datos de usuarios, por lo que
        debe tenerse en cuenta.
        """
        if config_name not in CONFIGS:
            raise ValueError(f"Configuración desconocida: {config_name}")

        cfg = CONFIGS[config_name]

        print(f"Cargando configuración: {config_name}")

        # Vectorizador
        embedder_class = cfg["embedder"]
        embedder = embedder_class()

        from ai_engine.services.vectorizer_service import VectorizerService
        self.vectorizer = VectorizerService(embedder)

        # Parámetros
        self.threshold = cfg["threshold"]
        self.DUPLICATE_THRESHOLD = cfg["duplicate_threshold"]
        self.dim = cfg["dim"]

        # Reiniciar / recargar index
        print("Reiniciando índice FAISS con dimensión:", self.dim)
        from ai_engine.services.user_indexer_service import CompositeIndexerService
        self.indexer = CompositeIndexerService(dim=self.dim, data_class=AuthUser)

        self.current_config = config_name

    def _save(self):
        """Guarda el índice FAISS en disco"""
        self.indexer.save(self.index_dir)

    def create_user(self, username: str, password: str, name: str, 
                   email: str, face_image: Image.Image) -> Tuple[AuthUser, str]:
        """
        Crea un nuevo usuario con reconocimiento facial.
        
        Args:
            username: Nombre de usuario
            password: Contraseña (luego se hashea)
            name: Nombre completo
            email: Email
            face_image: Imagen del rostro
        
        Returns:
            Tuple[AuthUser, str]: Usuario creado y path de la imagen (debug)
        
        Raises:
            ValueError: Si el usuario ya existe o validaciones fallan
        """
        from ai_engine.services.user_indexer_service import CompositeRecord
        
        
        # Preparar datos del usuario
        user_data = {
            'username': username,
            'name': name,
            'email': email,
            'password_hash': generate_password_hash(password),
            'registered_at': datetime.now().isoformat()
        }
        face_embedding = self.get_embedding(face_image)

        # Verifica si existe otro usuario con rostro similar, para evitar duplicados, es una comprobación muy básica
        duplicate = self.check_face_exists(face_embedding)
        if duplicate:
            raise ValueError(
                    f"Este rostro ya está registrado por el usuario '{duplicate['username']}' "
                    f"con similitud {duplicate['similarity']:.4f}"
                )

        # Guardar imagen sin procesar en el disco, sólo para debugging
        filename = f"{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        image_path = os.path.join(self.upload_dir, filename)
        face_image.save(image_path)

        user_data['face_image_path'] = image_path

        user = AuthUser(**user_data)

        # Agregar al índice FAISS
        record = [
            CompositeRecord(data=user, vector=face_embedding)
        ]
        self.indexer.add(record)
        
        # Guardar cambios en disco
        self._save()

        print(f"Usuario '{username}' creado exitosamente")
        return user, image_path


    def authenticate(self, face_image: Image.Image) -> Tuple[Optional[AuthUser], str, float]:
        """
        Autentica usuario con reconocimiento facial.
        
        Args:
            face_image: Imagen del rostro
        
        Returns:
            Tuple[Optional[AuthUser], str, float]: (usuario, mensaje, confianza)
        """
        
        # Obtener embedding
        try:
            face_embedding = self.get_embedding(face_image)
        except Exception as e:
            return None, f"Error al procesar rostro: {str(e)}", 0.0
        
        # Buscar usuario más similar, en este caso 2
        results = self.indexer.search(face_embedding, k=2)
        
        if not results:
            return None, "No hay usuarios registrados con reconocimiento facial", 0.0
        
        # Filtrar usuarios sin rostro, no debería ocurrir, pero por si acaso.
        valid_results = [
            (user, sim) for user, sim in results 
            if user.face_image_path is not None
        ]
        
        if not valid_results:
            return None, "No hay usuarios con reconocimiento facial configurado", 0.0
        
        best_user, best_similarity = valid_results[0]
        
        # Detectar matches ambiguos, si la distancia entre el mejor y el segundo es muy pequeña
        if len(valid_results) > 1:
            second_similarity = valid_results[1][1]
            if best_similarity - second_similarity < 0.05:
                return None, "Coincidencia ambigua. Intente de nuevo con mejor iluminación", 0.0
        
        # Verificar threshold
        if best_similarity < self.threshold:
            return None, f"No se encontró coincidencia (similitud: {best_similarity:.4f})", best_similarity
        
        # Login exitoso
        self._save()
        return best_user, "Login exitoso", best_similarity

    def check_face_exists(self, embedding: np.ndarray, threshold: Optional[float] = None) -> Optional[Dict]:
        """Verifica si un rostro ya está registrado"""
        if threshold is None:
            threshold = self.DUPLICATE_THRESHOLD
        
        results = self.indexer.search(embedding, k=1)
        
        if not results:
            return None
        
        user, similarity = results[0]
        
        # Ignorar usuarios sin rostro
        if user.face_image_path is None:
            return None
        
        if similarity >= threshold:
            return {
                'username': user.username,
                'similarity': float(similarity),
                'name': user.name
            }
        
        return None
    
    def get_stats(self) -> Dict:
        """Retorna estadísticas del sistema"""
        all_users = self.indexer.get_all_data()
        users_with_face = sum(1 for u in all_users if u.face_image_path is not None)
        
        return {
            'total_users': len(all_users),
            'users_with_face_auth': users_with_face,
            'index_path': str(self.index_dir),
            'threshold': self.threshold
        }

    def get_user(self, username: str) -> Optional[AuthUser]:
        for user in self.indexer.get_all_data():
                if user.username == username:
                    return user
        return None
    
    def get_embedding(self, image):
        return self.vectorizer.vectorize(image).flatten()