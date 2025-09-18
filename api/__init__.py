from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)

    from .routes.embeddings import embeddings_bp

    app.register_blueprint(embeddings_bp, url_prefix="/api/embeddings")

    return app