import json
import os

import cv2
import requests
from ultralytics import YOLO

import gradio as gr

# --- CONFIGURACIÓN ---
# ¡IMPORTANTE! Asegúrate de que esta URL coincida con la URL de tu API de Flask
API_ENDPOINT = os.environ.get("API_ENDPOINT", "http://localhost:1037/process_image/")

model = YOLO("yolo11n.pt")


# --- FUNCIÓN DE PROCESAMIENTO DE IMAGEN ---
def process_single_image(image_input_path):
    """
    Procesa una única imagen, enviándola como un archivo a una API externa
    y dibujando los datos de OCR (o similar) recibidos antes de mostrar la imagen.
    """
    if image_input_path is None:
        gr.Warning("Por favor, sube una imagen primero.")
        return None  # Devuelve None si no hay imagen

    # Leer la imagen usando OpenCV
    frame = cv2.imread(image_input_path)

    # --- VALIDACIÓN IMPORTANTE ---
    # Asegúrate de que la imagen se haya leído correctamente
    if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
        gr.Error(
            "Error al leer la imagen. Asegúrate de que el archivo es válido o no está dañado."
        )
        return None  # Devuelve None si la imagen es inválida

    gr.Info("Enviando imagen para procesamiento... Esto puede tardar un poco.")

    # Codificar la imagen a JPEG como bytes para enviarla como archivo
    _, img_encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    # Preparar la imagen para ser enviada como un archivo multipart/form-data
    # 'image' debe coincidir con el nombre del campo que tu API de Flask espera (request.files["image"])
    files = {"image": ("input_image.jpg", img_encoded.tobytes(), "image/jpeg")}

    # Usamos una copia para dibujar, manteniendo el original inalterado si hay errores
    processed_image = frame.copy()

    try:
        # Enviar la imagen a la API de procesamiento como un archivo
        response = requests.post(API_ENDPOINT, files=files, timeout=15)
        response.raise_for_status()  # Lanza un error para códigos de estado HTTP 4xx/5xx
        data_from_api = response.json()

        # Manejar errores devueltos por la API (si 'status' es 'error')
        if data_from_api.get("status") == "error":
            gr.Warning(
                f"API retornó un error: {data_from_api.get('message', 'Error desconocido de la API')}"
            )
            return (
                processed_image  # Devuelve la imagen original si hay un error de la API
            )

        # Dibujar sobre la imagen con los datos recibidos de la API
        # Asumimos que 'command_output' es una lista de objetos con 'bbox' y 'text'
        if "command_output" in data_from_api and isinstance(
            data_from_api["command_output"], list
        ):
            if not data_from_api["command_output"]:
                print("No se encontraron resultados en 'command_output'.")
            else:
                results = model(processed_image, verbose=False)

                bigger, area = [], 0
                for bbox in results[0].boxes.xyxy:
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    cv2.rectangle(processed_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # calculate the more biggest bbox
                    current_area = (x2 - x1) * (y2 - y1)
                    if current_area > area:
                        area = current_area
                        bigger = [x1, y1, x2, y2]

                if bigger:
                    x1, y1, x2, y2 = bigger
                    cv2.rectangle(processed_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    item = data_from_api["command_output"][0]
                    license = item[0]
                    accuracy = float(item[1])

                    cv2.putText(
                        processed_image,
                        f"License: {license} ({accuracy:.2f}%)",
                        (x1, y2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 0),
                        2,
                    )

        else:
            print(
                "La API no retornó 'command_output' o no está en el formato esperado."
            )

        if "command_error" in data_from_api and data_from_api["command_error"]:
            print(f"API Error: {data_from_api['command_error']}")

        gr.Info("Procesamiento de imagen completado.")

        # --- CONVERSIÓN CRÍTICA: BGR a RGB antes de devolver la imagen ---
        return cv2.cvtColor(
            processed_image, cv2.COLOR_BGR2RGB
        )  # Devuelve la imagen modificada

    except requests.exceptions.Timeout:
        gr.Warning(
            f"La solicitud a la API excedió el tiempo de espera. La imagen no fue procesada."
        )
        return processed_image  # Devuelve la imagen original
    except requests.exceptions.RequestException as e:
        gr.Error(
            f"Error de conexión con la API: {e}. Asegúrate de que tu API de Flask esté corriendo y sea accesible."
        )
        return processed_image  # Devuelve la imagen original
    except json.JSONDecodeError:
        gr.Error(
            f"Error al decodificar la respuesta JSON de la API. Respuesta inesperada o no es JSON."
        )
        return processed_image  # Devuelve la imagen original
    except Exception as e:
        gr.Error(f"Ocurrió un error inesperado al procesar la imagen: {e}")
        return processed_image  # Devuelve la imagen original


# --- INTERFAZ DE GRADIO ---
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 📸 **Procesamiento de Imagen con OCR** 🤖
        Sube una imagen para procesarla. La imagen será enviada a una API externa 
        (tu servidor Flask) para un procesamiento (ej. OCR), que devolverá los datos encontrados. 
        Estos datos se usarán para dibujar directamente en la imagen, mostrando el resultado.
        """
    )
    gr.Markdown("---")

    with gr.Row():
        # Componente para subir una imagen
        image_input = gr.Image(
            type="filepath", label="Sube tu imagen aquí", interactive=True
        )
        # Componente para mostrar la imagen procesada
        image_output = gr.Image(label="Imagen Procesada", interactive=False)

    process_button = gr.Button("🚀 Procesar Imagen")

    process_button.click(
        fn=process_single_image,
        inputs=image_input,
        outputs=image_output,
    )

    gr.Markdown(
        """
        ---
        **Instrucciones para la Puesta en Marcha:**
        1.  **Asegúrate de que tu API de Flask** esté funcionando y sea accesible. Por ejemplo, si se ejecuta en tu máquina local, debería estar en `http://localhost:1037/process_image/`. El código de Flask debe recibir la imagen como un **archivo** usando `request.files["image"]`.
        2.  **Sube un archivo de imagen** (JPEG, PNG, etc.) usando el componente "Sube tu imagen aquí".
        3.  Haz clic en el botón **'Procesar Imagen'** para ver el resultado con las detecciones dibujadas.
        """
    )

demo.launch()
