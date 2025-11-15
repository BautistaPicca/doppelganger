from pathlib import Path
from flask import Flask
from flask_cors import CORS

from api.services.auth_service import FaceAuthService

def create_app():
    app = Flask(__name__)
    CORS(app)
    app.config['SECRET_KEY'] = 'malloc'
    app.config["UPLOAD_FOLDER"] = "run/server/upload"
    app.config["FACE_INDEX_DIR"] = "run/server/face_index"
    app.config["FACE_THRESHOLD"] = 0.65

    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["FACE_INDEX_DIR"]).mkdir(parents=True, exist_ok=True)

    import api.config as cfg
    cfg.auth_service = FaceAuthService(
        upload_dir=app.config["UPLOAD_FOLDER"],
        index_dir=app.config["FACE_INDEX_DIR"],
        config_name="pretrained"
    )
    
    from api.routes.match import match_bp
    from api.routes.auth import auth_bp
    from api.routes.config import config_bp
    app.register_blueprint(match_bp, url_prefix="/api/match")
    app.register_blueprint(auth_bp, url_prefix="/api/auth")
    app.register_blueprint(config_bp, url_prefix="/api")
    
    return app

app = create_app()