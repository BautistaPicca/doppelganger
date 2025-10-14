from flask import Blueprint, request, jsonify
from PIL import Image
from io import BytesIO

from ai_engine.matcher.service import MatcherService

embeddings_bp = Blueprint("embeddings", __name__)
matcher = MatcherService(index_dir="run/index")

@embeddings_bp.route("/", methods=["POST"])
def embed():
    if "image" not in request.files:
        return jsonify({"error": "No se envió imagen para comparar"}), 400
    
    file = request.files["image"]
    try:
        image = Image.open(BytesIO(file.read()))
        results = matcher.match_image(image, k=5)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500