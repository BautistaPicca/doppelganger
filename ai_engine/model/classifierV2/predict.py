import torch
import cv2
from torchvision import transforms
from ai_engine.implementations.detector.face_detector import detect_and_crop_face


#Metodo para realizar la prediccion
def predict_celebrity(model, class_names, image_path, device, img_size):

    face = detect_and_crop_face(image_path)

    if face is None:
        return None, 0.0

    # Convertir BGR -> RGB si es necesario
    if face.shape[2] == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = transform(face).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        idx = torch.argmax(probs).item()

    return class_names[idx], probs[idx].item()
