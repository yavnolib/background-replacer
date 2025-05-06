import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def download():
    logger.info("Downloading data and model files using DVC...")
    result = subprocess.run(["dvc", "pull"], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("DVC pull failed:\n%s", result.stderr)
        raise RuntimeError("DVC download failed")
    logger.info("DVC pull completed successfully.")

if __name__ == "__main__":
    download()
