from flask import Blueprint, request, jsonify
from PIL import Image
from io import BytesIO

from ai_engine.services.classifier_service import ClassifierService

match_bp = Blueprint("match", __name__)

classifier_service = ClassifierService(
    model_path="run/models/best_model.pth",
    mapping_path="run/models/celebrity_mapping.json",
    device="cuda" if False else "cpu"
)

@match_bp.route("/", methods=["POST"])
def match():
    if "image" not in request.files:
        return jsonify({"error": "No se envió imagen para comparar"}), 400
    
    file = request.files["image"]
    try:
        image = Image.open(BytesIO(file.read()))
        results = classifier_service.predict(image, top_k=3)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500