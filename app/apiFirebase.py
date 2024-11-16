import firebase_admin
from firebase_admin import credentials, storage
import datetime
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from cargarModelo import preprocess_image, predict, plot_image_with_boxes
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO

app = FastAPI()

# Habilitar CORS para todos los orígenes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos
    allow_headers=["*"],  # Permite todos los encabezados
)

# Inicializa Firebase Admin SDK
cred = credentials.Certificate(
    "macrofitas-8bbaa-firebase-adminsdk-glls0-91601c224b.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'macrofitas-8bbaa.firebasestorage.app'
})
bucket = storage.bucket()


@app.post("/upload-image/{user_uid}")
# Asegúrate de recibir el UID del usuario
async def upload_image(user_uid: str, file: UploadFile = File(...)):
    # Leer la imagen cargada en memoria
    file_content = await file.read()

    # Guardar temporalmente en memoria para preprocesamiento
    tmp_buffer = BytesIO(file_content)
    tmp_buffer.seek(0)

    # Preprocesar la imagen y realizar predicciones
    img_tensor = preprocess_image(tmp_buffer)
    prediction = predict(img_tensor)

    # Crear nombre descriptivo para la imagen con las cajas detectadas
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"{timestamp}_{file.filename}"

    # Llamar a la función plot_image_with_boxes para generar la imagen procesada en memoria
    output_buffer = plot_image_with_boxes(tmp_buffer, prediction)
    output_buffer.seek(0)  # Reiniciar el puntero del buffer

    # Crear una referencia en Firebase Storage con el UID del usuario
    # Ruta con el UID del usuario
    blob = bucket.blob(f"users/{user_uid}/images/{output_filename}")
    # Sube el archivo generado a Firebase Storage
    blob.upload_from_file(output_buffer, content_type="image/png")
    output_buffer.close()  # Cierra el buffer para liberar recursos

    # Haz que el archivo sea público
    blob.make_public()

    # Generar una URL pública para acceder a la imagen
    image_url = blob.public_url

    # Devuelve un JSON con el nombre de la imagen y la URL pública
    return JSONResponse(content={
        "message": "Image processed and uploaded successfully",
        "image_path": image_url
    })


@app.get("/get-image/{user_uid}/{image_name}")
async def get_image(user_uid: str, image_name: str):
    # Buscar la imagen en Firebase Storage con el UID del usuario
    blob = bucket.blob(f"users/{user_uid}/images/{image_name}")

    if blob.exists():  # Verifica si el archivo existe en el bucket
        # Obtiene la URL pública de la imagen
        image_url = blob.public_url
        return JSONResponse(content={"image_url": image_url})
    # Si la imagen no existe, devuelve un JSON con el error
    return JSONResponse(status_code=404, content={"message": "Image not found"})


@app.delete("/delete-image/{user_uid}/{image_name}")
async def delete_image(user_uid: str, image_name: str):
    # Crear la referencia a la imagen en Firebase Storage
    blob = bucket.blob(f"users/{user_uid}/images/{image_name}")

    # Verificar si el archivo existe en el bucket
    if blob.exists():
        # Eliminar el archivo del bucket
        blob.delete()
        return JSONResponse(content={"message": "Image deleted successfully"})

    # Si el archivo no existe, devolver un error 404
    return JSONResponse(status_code=404, content={"message": "Image not found"})
