from flask import Blueprint, request, jsonify

embeddings_bp = Blueprint("embeddings", __name__)

@embeddings_bp.route("/", methods=["POST"])
def embed():
    image = request.files["image"]
    return jsonify({"embedding": "Not implemented yet"})