import json
import requests
import os

# --- CONFIGURACIÓN ---
# ¡IMPORTANTE! Asegúrate de que esta URL coincida con la URL de tu API de Flask
API_ENDPOINT = "http://localhost:1037/process_image/" 

# Ruta a la imagen que quieres enviar
# Asegúrate de que esta imagen exista en tu disco.
# Puedes cambiarla por cualquier otra imagen de prueba.
IMAGE_PATH = "test/rojo_4.png" # <--- ¡CAMBIA ESTO POR LA RUTA REAL DE TU IMAGEN!

def send_image_to_api(image_path):
    """
    Envía una imagen local a la API de procesamiento de Flask
    y muestra la respuesta.
    """
    if not os.path.exists(image_path):
        print(f"Error: La imagen '{image_path}' no se encontró.")
        return

    print(f"Enviando imagen: {image_path} a {API_ENDPOINT}")

    try:
        # Abrir la imagen en modo binario para enviarla como archivo
        with open(image_path, 'rb') as f:
            # 'image' debe coincidir con el nombre del campo que tu API de Flask espera (request.files["image"])
            # ('nombre_archivo.jpg', contenido_bytes, 'tipo_mime')
            files = {'image': (os.path.basename(image_path), f.read(), 'image/jpeg')}
            
            # Enviar la solicitud POST
            response = requests.post(API_ENDPOINT, files=files, timeout=10)
            
            # Lanzar una excepción si la solicitud no fue exitosa (código de estado 4xx o 5xx)
            response.raise_for_status()

            # Obtener y mostrar la respuesta JSON de la API
            data_from_api = response.json()
            print("\n--- Respuesta de la API ---")
            print(data_from_api)

            # Puedes añadir lógica aquí para procesar la 'command_output' si lo deseas
            if data_from_api.get("status") == "success" and "command_output" in data_from_api:
                print("\nDetecciones/Salida de OCR:")
                for item in data_from_api["command_output"]:
                    print(f"  - BBox: {item.get('bbox')}, Texto: '{item.get('text')}'")
            elif data_from_api.get("status") == "error":
                print(f"\nLa API retornó un error: {data_from_api.get('message', 'Error desconocido')}")

    except requests.exceptions.Timeout:
        print(f"Error: La solicitud excedió el tiempo de espera ({10} segundos).")
    except requests.exceptions.RequestException as e:
        print(f"Error de conexión con la API: {e}. Asegúrate de que tu API de Flask esté corriendo y sea accesible.")
    except json.JSONDecodeError:
        print("Error: La respuesta de la API no es un JSON válido.")
        print("Respuesta de la API (texto crudo):", response.text)
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    # Llama a la función para enviar la imagen
    send_image_to_api(IMAGE_PATH)