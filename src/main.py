import logging
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

warnings.filterwarnings("ignore")

from join_nuclei_and_cilia_results import main as join_nuclei_and_cilia_results
from run_cilia_detection import main as run_cilia_detection
from run_nuclei_segmentation import main as run_nuclei_segmentation
from utils import compress_directory_to_zip, setup_logging

load_dotenv("env.conf")
setup_logging()

logger = logging.getLogger("main")


def _check_directory(path: Path):
    """Checks if the given path exists and is a directory.

    Args:
        path: Path to check.

    Raises:
        RuntimeError: If the directory does not exist or the path is not a directory.
    """
    if not path.exists():
        logger.error(f"Directory not found: '{path}'")
        raise RuntimeError(f"Directory not found: '{path}'")
    if not path.is_dir():
        logger.error(f"Path is not a directory: '{path}'")
        raise RuntimeError(f"Path is not a directory: '{path}'")


def main():
    """Main function to execute cilia detection, nuclei segmentation, merge results, and compress the output directory."""
    images_dir = Path(os.environ.get("IMAGES_DIR"))
    output_dir = Path(os.environ.get("OUTPUT_DIR"))
    logger.info(f"Processing data in {images_dir}")
    try:
        _ = _check_directory(images_dir)
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
