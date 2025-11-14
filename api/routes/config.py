from flask import Blueprint, request, jsonify

from api.config import auth_service

config_bp = Blueprint("api", __name__)

@config_bp.route("/config", methods=["POST"])
def change_config():
    data = request.get_json()

    mode = data.get("mode")

    try:
        auth_service.apply_config(mode)
        return jsonify({"message": f"Configuración cambiada a {mode}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400