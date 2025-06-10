# Usa una imagen base de Python
FROM python:3.10-slim-buster

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instala dependencias del sistema necesarias para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    ffmpeg \
    libglib2.0-0 
    # Clean up apt cache to reduce image size
RUN rm -rf /var/lib/apt/lists/*


# Copia los archivos de requisitos e instálalos primero para aprovechar el cache de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código de la aplicación de Gradio
COPY api.py .

# Expone el puerto que Gradio va a usar
EXPOSE 7860

# Comando para ejecutar la aplicación de Gradio
# Gradio por defecto usa el puerto 7860 y escucha en 0.0.0.0 cuando se lanza.
CMD ["python", "api.py"]
