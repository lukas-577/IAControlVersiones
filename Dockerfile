# Usar una imagen base de Python
FROM python:3.9-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos de tu proyecto al contenedor
COPY . /app

# Instalar las dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto en el que la app estará corriendo
EXPOSE 8080

# Comando para ejecutar tu aplicación con Uvicorn
CMD ["uvicorn", "apiFirebase:app", "--host", "0.0.0.0", "--port", "8080"]
