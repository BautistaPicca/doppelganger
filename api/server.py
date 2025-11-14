from pathlib import Path
from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)
    app.config['SECRET_KEY'] = 'malloc'
    app.config["UPLOAD_FOLDER"] = "run/server/upload"
    app.config["FACE_INDEX_DIR"] = "run/server/face_index"
    app.config["FACE_THRESHOLD"] = 0.65
    
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["FACE_INDEX_DIR"]).mkdir(parents=True, exist_ok=True)
    
    from api.routes.embeddings import embeddings_bp
    from api.routes.auth import auth_bp
    app.register_blueprint(embeddings_bp, url_prefix="/embed")
    app.register_blueprint(auth_bp, url_prefix="/api/auth")

    return app

app = create_app()