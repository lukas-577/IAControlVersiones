import firebase_admin
from firebase_admin import credentials, storage
import datetime
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from cargarModelo import preprocess_image, predict, plot_image_with_boxes
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import os
import uvicorn


# Descripción general de la API
description = """
### API de Gestión de Imágenes

Esta API permite a los usuarios realizar las siguientes acciones:

1. **Subir imágenes**: Puedes cargar imágenes, procesarlas y almacenarlas en Firebase Storage.
2. **Obtener imágenes**: Consulta las URL públicas de las imágenes subidas.
3. **Eliminar imágenes**: Borra imágenes específicas del almacenamiento.

#### Cómo usar la API
1. Usa el endpoint `POST /upload-image/{user_uid}` para subir imágenes. Asegúrate de proporcionar el `user_uid` y el archivo.
2. Usa `GET /get-image/{user_uid}/{image_name}` para recuperar la URL pública de una imagen.
3. Usa `DELETE /delete-image/{user_uid}/{image_name}` para eliminar una imagen específica.

#### Notas importantes
- Asegúrate de que el `user_uid` sea válido.
- Los archivos deben ser de tipo imagen.
- Consulta la documentación de cada endpoint para más detalles.

"""


app = FastAPI(
    title="API de Gestión de Imágenes IA con Firebase Storage",
    description=description,
    version="1.0.0",
    contact={
        "name": "API IA con Firebase Storage",
        "email": "lmedinar@utem.cl",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

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


@app.get("/", tags=["General"])
def read_root():
    """Verifica que el servicio esté operativo."""
    return {"message": "Todo ok"}


@app.post(
    "/upload-image/{user_uid}",
    tags=["Gestión de Imágenes"],
    summary="Subir imagen",
    description="Sube una imagen, procesa la misma y la guarda en Firebase Storage. Genera una URL pública para la imagen procesada.",
)
# Asegúrate de recibir el UID del usuario
async def upload_image(user_uid: str, file: UploadFile = File(...)):
    """Sube una imagen, procesa y la guarda en Firebase Storage.

    - **user_uid**: ID del usuario.
    - **file**: Imagen a subir.
    """

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


@app.get(
    "/get-image/{user_uid}/{image_name}",
    tags=["Gestión de Imágenes"],
    summary="Obtener imagen",
    description="Obtiene la URL pública de una imagen almacenada en Firebase Storage.",
)
async def get_image(user_uid: str, image_name: str):
    """Obtiene una imagen almacenada.

    - **user_uid**: ID del usuario.
    - **image_name**: Nombre del archivo.
    """
    # Buscar la imagen en Firebase Storage con el UID del usuario
    blob = bucket.blob(f"users/{user_uid}/images/{image_name}")

    if blob.exists():  # Verifica si el archivo existe en el bucket
        # Obtiene la URL pública de la imagen
        image_url = blob.public_url
        return JSONResponse(content={"image_url": image_url})
    # Si la imagen no existe, devuelve un JSON con el error
    return JSONResponse(status_code=404, content={"message": "Image not found"})


@app.delete(
    "/delete-image/{user_uid}/{image_name}",
    tags=["Gestión de Imágenes"],
    summary="Eliminar imagen",
    description="Elimina una imagen específica almacenada en Firebase Storage.",
)
async def delete_image(user_uid: str, image_name: str):
    """Elimina una imagen específica.

    - **user_uid**: ID del usuario.
    - **image_name**: Nombre del archivo.
    """

    # Crear la referencia a la imagen en Firebase Storage
    blob = bucket.blob(f"users/{user_uid}/images/{image_name}")

    # Verificar si el archivo existe en el bucket
    if blob.exists():
        # Eliminar el archivo del bucket
        blob.delete()
        return JSONResponse(content={"message": "Image deleted successfully"})

    # Si el archivo no existe, devolver un error 404
    return JSONResponse(status_code=404, content={"message": "Image not found"})

# Asegúrate de que el servidor se inicie correctamente
if __name__ == "__main__":
    # Obtiene el puerto desde la variable de entorno
    port = int(os.environ.get("PORT", 8080))
    print(f"Servidor iniciado en el puerto {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
