import torch
from ai_engine.model.load_Model import load_model
from ai_engine.utils.classes import load_class_names
from ai_engine.model.classifierV2.predict import predict_celebrity


from ai_engine.utils import MODEL_PATH, DATA_DIR, IMG_SIZE

TEST_IMAGE = r"ruta de ejemplo"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    class_names = load_class_names(DATA_DIR)
    model = load_model(MODEL_PATH, len(class_names), device)

    celeb, prob = predict_celebrity(
        model=model,
        class_names=class_names,
        image_path=TEST_IMAGE,
        device=device,
        img_size=IMG_SIZE
    )

    if celeb is None:
        print("No se pudo predecir.")
    else:
        print(f"Predicción: {celeb}")
        print(f"Confianza: {prob * 100:.2f}%")

if __name__ == "__main__":
    main()
