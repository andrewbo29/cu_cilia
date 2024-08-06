from dotenv import load_dotenv
import logging
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from join_nuclei_and_cilia_results import main as join_nuclei_and_cilia_results
from utils import compress_directory_to_zip, setup_logging
from run_cilia_detection import main as run_cilia_detection
from run_nuclei_segmentation import main as run_nuclei_segmentation


load_dotenv("env.txt")
setup_logging()

logger = logging.getLogger("main")

def _check_directory(path: Path):
    if not path.exists():
        logger.error(f"Directory not found: '{path}'")
        raise RuntimeError(f"Directory not found: '{path}'")
    if not path.is_dir():
        logger.error(f"Path is not a directory: '{path}'")
        raise RuntimeError(f"Path is not a directory: '{path}'")

    return len(list(path.iterdir()))


def main():
    images_dir = Path(os.environ.get("IMAGES_DIR"))
    output_dir = Path(os.environ.get("OUTPUT_DIR"))
    logger.info(f"Processing data in {images_dir}")
    try:
        n_files = _check_directory(images_dir)
        logger.info(f"Found {n_files} images to process")
        logger.info(f"Saving results to {output_dir}")
        run_cilia_detection()
        run_nuclei_segmentation()
        join_nuclei_and_cilia_results()
        logger.info("Zipping results")
        compress_directory_to_zip(output_dir)
    except RuntimeError as e:
        return


if __name__ == "__main__":
    main()
