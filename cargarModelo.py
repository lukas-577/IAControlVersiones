import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
import torch
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from io import BytesIO

# Cargar el modelo preentrenado
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = 7  # Fondo y planta

# Modificar la capa de clasificación
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Cargar los pesos entrenados
model.load_state_dict(torch.load(
    'fasterrcnn_planta_v2.pth', map_location=torch.device('cpu')))
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


# Diccionario de clases (ajusta los nombres según tus clases)
class_names = {
    0: 'Fondo',
    1: 'A_capillaris',
    2: 'Egeria_densa',
    3: 'Ludwigia_peploides',
    4: 'Myriophyllum_aquaticum',
    5: 'Nasturtium_officinale',  # Creo que esta no la esta tomando
    6: 'Veronica_Anagallis_aquaticum',
}


# Dibuja rectangulos


def plot_image_with_boxes(image_path, prediction):
    # Cargar la imagen original
    img = Image.open(image_path).convert("RGB")

    # Crear la figura
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Obtener las cajas detectadas y las puntuaciones
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()

    # Dibujar un rectángulo para cada caja detectada con puntuación mayor a un umbral
    for i, box in enumerate(boxes):
        if scores[i] > 0.2:  # Umbral de puntuación
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            # Añadir el nombre de la clase
            class_name = class_names.get(labels[i], 'Desconocido')
            plt.text(x_min, y_min - 10, class_name, color='red',
                     fontsize=10, backgroundcolor="white")

    # Guardar la imagen procesada en un buffer
    buffer = BytesIO()
    fig.savefig(buffer, format="PNG")
    buffer.seek(0)  # Reiniciar el puntero del buffer
    plt.close(fig)  # Cerrar la figura para liberar memoria

    return buffer  # Retorna el buffer en lugar de guardar un archivo
