"""Command-line client to send images to open_alpr_api LitServe endpoint."""

import argparse
import base64
import json
import os
import sys
from typing import Optional

import requests

DEFAULT_ENDPOINT = os.getenv("API_ENDPOINT", "http://localhost:8000/predict")


def send_image_to_api(
    image_path: str,
    endpoint: str = DEFAULT_ENDPOINT,
    output_path: Optional[str] = None,
    timeout: float = 15.0,
) -> int:
    """Send an image to the ALPR API endpoint and print structured results.

    Args:
        image_path: Path to the image file to send.
        endpoint: HTTP URL of the LitServe API endpoint.
        output_path: Optional path to write annotated image.
        timeout: HTTP request timeout in seconds.

    Returns:
        Exit code: 0 on success, non-zero on failure.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found.", file=sys.stderr)
        return 1

    print(f"[*] Target Endpoint: {endpoint}")
    print(f"[*] Sending image:   {image_path}")

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        files = {"image": (os.path.basename(image_path), image_bytes, "image/jpeg")}
        data = {"annotate": "true"} if output_path else {}

        response = requests.post(endpoint, files=files, data=data, timeout=timeout)
        response.raise_for_status()
        payload = response.json()

        print("\n================== API RESPONSE ==================")
        print(json.dumps(payload, indent=2))
        print("==================================================")

        license_info = payload.get("license_plate", {})
        plate = license_info.get("plate", "N/A")
        confidence = license_info.get("confidence")
        conf_str = f"{confidence:.2f}%" if confidence is not None else "N/A"
        latency_ms = payload.get("latency_ms", 0.0)
        vehicle_count = len(payload.get("all_vehicles", []))

        print("\n--- Summary ---")
        print(f"Detected Plate:    {plate}")
        print(f"Confidence:        {conf_str}")
        print(f"Vehicles Found:    {vehicle_count}")
        print(f"Server Latency:    {latency_ms:.2f} ms")

        # Save annotated image if requested and returned
        b64_img = payload.get("annotated_image_base64")
        if output_path and b64_img:
            img_data = base64.b64decode(b64_img)
            with open(output_path, "wb") as out_file:
                out_file.write(img_data)
            print(f"[+] Saved annotated image to: {output_path}")

        return 0

    except requests.exceptions.Timeout:
        print(f"Error: Request timed out after {timeout} seconds.", file=sys.stderr)
        return 2
    except requests.exceptions.RequestException as exc:
        print(f"Connection error: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 4


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CLI tool to test open_alpr_api high-performance inference."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="test/rojo_4.png",
        help="Path to the input image file (default: test/rojo_4.png)",
    )
    parser.add_argument(
        "--endpoint",
        "-e",
        default=DEFAULT_ENDPOINT,
        help=f"API URL endpoint (default: {DEFAULT_ENDPOINT})",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional path to save annotated output image",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=float,
        default=15.0,
        help="Request timeout in seconds (default: 15.0)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(
        send_image_to_api(
            image_path=args.image,
            endpoint=args.endpoint,
            output_path=args.output,
            timeout=args.timeout,
        )
    )
