from flask import Flask
from api.routes.embeddings import embeddings_bp
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(embeddings_bp, url_prefix="/embed")

    return app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
