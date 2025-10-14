from flask import Flask
from api.routes.embeddings import embeddings_bp

def create_app():
    app = Flask(__name__)
    app.register_blueprint(embeddings_bp, url_prefix="/embed")

    return app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
