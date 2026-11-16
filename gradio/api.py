"""Gradio Web UI frontend for open_alpr_api."""

import base64
import os
from typing import Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import requests

API_ENDPOINT = os.environ.get("API_ENDPOINT", "http://api:8000/predict")


def process_image(
    image_input: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], str, str, str]:
    """Send image array to LitServe ALPR API and return annotated image with metadata.

    Args:
        image_input: NumPy RGB array provided by Gradio.

    Returns:
        Tuple of (annotated_rgb_image, plate_display_text, confidence_text, latency_text).
    """
    if image_input is None:
        gr.Warning("Please upload an image first.")
        return None, "N/A", "N/A", "0.0 ms"

    # Convert RGB from Gradio to BGR for OpenCV encoding
    image_bgr = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
    success, img_encoded = cv2.imencode(
        ".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    )
    if not success:
        gr.Error("Failed to encode image to JPEG.")
        return image_input, "Encoding Error", "N/A", "0.0 ms"

    # Send multipart request with annotate=True flag
    files = {"image": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
    data = {"annotate": "true"}

    try:
        response = requests.post(API_ENDPOINT, files=files, data=data, timeout=20)
        response.raise_for_status()
        payload = response.json()

        license_info = payload.get("license_plate", {})
        plate = license_info.get("plate") or "Not Detected"
        conf = license_info.get("confidence")
        conf_str = f"{conf:.2f}%" if conf is not None else "N/A"
        latency_str = f"{payload.get('latency_ms', 0.0):.2f} ms"

        # Check if server provided pre-annotated image
        b64_img = payload.get("annotated_image_base64")
        if b64_img:
            img_bytes = base64.b64decode(b64_img)
            nparr = np.frombuffer(img_bytes, np.uint8)
            annotated_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            return annotated_rgb, plate, conf_str, latency_str

        # Fallback local drawing if annotated image not returned
        annotated_bgr = image_bgr.copy()
        vehicles = payload.get("all_vehicles", [])
        for v in vehicles:
            bbox = v.get("bbox", [])
            if len(bbox) == 4:
                cv2.rectangle(
                    annotated_bgr,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    (0, 255, 0),
                    2,
                )

        if payload.get("primary_vehicle"):
            p_bbox = payload["primary_vehicle"].get("bbox", [])
            if len(p_bbox) == 4:
                cv2.rectangle(
                    annotated_bgr,
                    (p_bbox[0], p_bbox[1]),
                    (p_bbox[2], p_bbox[3]),
                    (0, 0, 255),
                    3,
                )
                label = f"{plate} ({conf_str})"
                cv2.putText(
                    annotated_bgr,
                    label,
                    (p_bbox[0], max(p_bbox[1] - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )

        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        return annotated_rgb, plate, conf_str, latency_str

    except requests.exceptions.Timeout:
        gr.Error("API request timed out after 20s.")
        return image_input, "Timeout", "N/A", "0.0 ms"
    except requests.exceptions.RequestException as exc:
        gr.Error(f"Connection error with API: {exc}")
        return image_input, "Connection Error", "N/A", "0.0 ms"
    except Exception as exc:
        gr.Error(f"Unexpected error: {exc}")
        return image_input, "Error", "N/A", "0.0 ms"


# --- Gradio UI Layout ---
with gr.Blocks(title="OpenALPR High-Performance API") as demo:
    gr.Markdown(
        """
        # 🚗 **OpenALPR + wyolo Enterprise Vision** ⚡
        High-throughput Automatic License Plate Recognition powered by **LitServe** and **wyolo** (TensorRT / FP16).
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="numpy", label="Input Image", interactive=True
            )
            submit_btn = gr.Button("⚡ Detect & Recognize", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(label="Annotated Detections", interactive=False)
            with gr.Row():
                plate_box = gr.Textbox(label="Detected License Plate", interactive=False)
                conf_box = gr.Textbox(label="Confidence", interactive=False)
                latency_box = gr.Textbox(label="Inference Latency", interactive=False)

    submit_btn.click(
        fn=process_image,
        inputs=[input_image],
        outputs=[output_image, plate_box, conf_box, latency_box],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
