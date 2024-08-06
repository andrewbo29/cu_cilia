import logging
import shutil
import os
from pathlib import Path


def setup_logging():
    level = logging.DEBUG if os.environ.get("DEBUG", "False") == "True" else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(name)s - %(levelname)s - %(message)s',
        handlers=[
            # logging.FileHandler("project.log"),
            logging.StreamHandler()
        ]
    )


def compress_directory_to_zip(output_dir: Path):
    """
    Compresses the given directory into a zip file.

    Args:
        output_dir: Path to the directory to compress.
        zip_filename: Name of the output zip file (stem).
    """
    if not output_dir.is_dir():
        raise ValueError(f"The directory '{output_dir}' does not exist or is not a directory.")

    shutil.make_archive(str(output_dir), 'zip', output_dir)
    if os.environ.get("DEBUG", "False") == "True":
        print(f"Directory '{output_dir}' has been compressed into '{output_dir}.zip'.")
