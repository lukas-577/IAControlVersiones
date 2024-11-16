# main.py
import datetime
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from cargarModelo import preprocess_image, predict, plot_image_with_boxes
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os

app = FastAPI()

# Habilitar CORS para todos los orígenes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos
    allow_headers=["*"],  # Permite todos los encabezados
)


# Define el directorio para guardar las imágenes
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Crear carpeta si no existe


@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    # Guardar temporalmente la imagen cargada
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Preprocesar la imagen y realizar predicciones
    img_tensor = preprocess_image(tmp_path)
    prediction = predict(img_tensor)

    # Crear nombre descriptivo para la imagen con las cajas detectadas
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"{timestamp}_{file.filename}"

    # Guardar la imagen con las cajas detectadas en la carpeta
    output_path = f"{UPLOAD_DIR}/{output_filename}"
    print(output_path)
    plot_image_with_boxes(tmp_path, prediction, output_path)

    # Devuelve un JSON con el nombre de la imagen
    return JSONResponse(content={"message": "Image processed successfully", "image_path": output_path})


@app.get("/get-image/{image_name}")
async def get_image(image_name: str):

    # Construye la ruta completa a la imagen
    image_path = os.path.join(UPLOAD_DIR, image_name)
    # Asegúrate de que la imagen esté en una ubicación accesible
    if os.path.exists(image_path):
        return FileResponse(image_path, media_type="image/jpeg")
    return JSONResponse(status_code=404, content={"message": "Image not found"})
