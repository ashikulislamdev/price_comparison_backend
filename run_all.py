import subprocess

mock_api = subprocess.Popen([
    "uvicorn", "mock_api.main:app",
    "--host", "0.0.0.0",  # to allow access from other devices
    "--port", "8000", "--reload"
])

price_api = subprocess.Popen([
    "uvicorn", "price_comparison_api.main:app",
    "--host", "0.0.0.0",  
    "--port", "8001", "--reload"
])

try:
    mock_api.wait()
    price_api.wait()
except KeyboardInterrupt:
    mock_api.terminate()
    price_api.terminate()