import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
import torch
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Cargar el modelo preentrenado
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = 2  # Fondo y planta

# Modificar la capa de clasificación
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Cargar los pesos entrenados
model.load_state_dict(torch.load('fasterrcnn_planta.pth'))
model.eval()  # Poner el modelo en modo de evaluación


# Procesa la imagen


# Definir una función para preprocesar la imagen

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])  # Convertir la imagen en un tensor
    return transform(img)


def predict(image_tensor):
    # Realiza la predicción
    with torch.no_grad():
        prediction = model([image_tensor])
    return prediction


# Dibuja rectangulos


def plot_image_with_boxes(image_path, prediction, output_path):
    # Cargar la imagen original
    img = Image.open(image_path).convert("RGB")

    # Crear la figura
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Obtener las cajas detectadas y las puntuaciones
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    # Dibujar un rectángulo para cada caja detectada con puntuación mayor a un umbral
    for i, box in enumerate(boxes):
        if scores[i] > 0.4:  # Umbral de puntuación
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    fig.savefig(output_path)
    # plt.show()
    plt.close(fig)
