"""Unit tests for OpenALPRClient.

This test module verifies the HTTP communication and parsing logic of OpenALPRClient,
including connection pooling, timeout handling, error reporting, and payload extraction.
Every test function includes detailed documentation explaining the scenario and assertions.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from app.client import OpenALPRClient


@pytest.fixture
def client() -> OpenALPRClient:
    """Fixture providing an OpenALPRClient configured with a mock endpoint."""
    return OpenALPRClient(endpoint="http://mock-alpr:8000/process_image/", timeout=5.0)


def test_openalpr_client_success_response(client: OpenALPRClient) -> None:
    """Validate successful parsing of OpenALPR backend response.

    What this test does:
    1. Simulates an HTTP 200 response from OpenALPR with multiple recognized plates.
    2. Calls `process_image_bytes` with dummy JPEG byte payload.
    3. Validates that:
       - The best candidate plate text matches the top item.
       - The confidence percentage is correctly parsed as a float.
       - The candidates list contains all items returned by the backend.
       - The error message is None.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "status": "success",
        "command_output": [
            ["1234ABC", "94.50"],
            ["1234AB0", "78.20"],
        ],
    }

    with patch.object(client.session, "post", return_value=mock_response) as mock_post:
        dummy_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        plate, conf, candidates, err = client.process_image_bytes(dummy_bytes)

        # Ensure POST was called once with multipart files
        mock_post.assert_called_once()
        assert plate == "1234ABC"
        assert conf == pytest.approx(94.50, rel=1e-2)
        assert len(candidates) == 2
        assert err is None


def test_openalpr_client_empty_command_output(client: OpenALPRClient) -> None:
    """Validate handling of an empty command_output list from backend.

    What this test does:
    1. Simulates a backend response where no license plates were detected in the image.
    2. Asserts that plate and conf are None, candidates list is empty, and error is None.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "status": "success",
        "command_output": [],
    }

    with patch.object(client.session, "post", return_value=mock_response):
        plate, conf, candidates, err = client.process_image_bytes(b"dummy_bytes")
        assert plate is None
        assert conf is None
        assert candidates == []
        assert err is None


def test_openalpr_client_backend_error_payload(client: OpenALPRClient) -> None:
    """Validate handling when the backend explicitly returns status='error'.

    What this test does:
    1. Simulates a response payload where status is 'error' and a message is supplied.
    2. Asserts that the error message is extracted and returned properly without raising.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "status": "error",
        "message": "Engine initialization failed",
    }

    with patch.object(client.session, "post", return_value=mock_response):
        plate, conf, candidates, err = client.process_image_bytes(b"dummy_bytes")
        assert plate is None
        assert conf is None
        assert "Engine initialization failed" in str(err)


def test_openalpr_client_timeout_handling(client: OpenALPRClient) -> None:
    """Validate graceful handling of network timeouts.

    What this test does:
    1. Simulates a `requests.exceptions.Timeout` raised during the HTTP POST request.
    2. Validates that the function catches the exception and returns a descriptive error string.
    """
    with patch.object(
        client.session, "post", side_effect=requests.exceptions.Timeout("Connection timed out")
    ):
        plate, conf, candidates, err = client.process_image_bytes(b"dummy_bytes")
        assert plate is None
        assert conf is None
        assert candidates == []
        assert err is not None
        assert "timed out" in err.lower()


def test_openalpr_client_http_error_handling(client: OpenALPRClient) -> None:
    """Validate handling of HTTP 500/502/503 status code errors.

    What this test does:
    1. Simulates an HTTP 502 Bad Gateway response raising `requests.exceptions.HTTPError`.
    2. Validates that the error is safely captured and reported in the returned tuple.
    """
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("502 Bad Gateway")

    with patch.object(client.session, "post", return_value=mock_response):
        plate, conf, candidates, err = client.process_image_bytes(b"dummy_bytes")
        assert plate is None
        assert conf is None
        assert err is not None
        assert "failed" in err.lower() or "502" in err
