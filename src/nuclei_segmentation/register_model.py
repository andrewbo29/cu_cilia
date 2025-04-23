import logging
import os
import sys
from pathlib import Path

import mlflow
from cellpose import models
from dotenv import load_dotenv

load_dotenv(".env")  # loading mlflow credentials
logger = logging.getLogger(__name__)


class NucleiSegModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model_path):
        super().__init__()
        self.model = models.CellposeModel(gpu=False, pretrained_model=model_path)


def register_model():
    mlflow.set_experiment(os.environ.get("MLFLOW_MODEL_NAME"))

    root_path = Path(__file__).parent.parent.parent.resolve()
    out_dir = root_path / "models"
    model_name = "cellpose_thyroid"
    model_path = out_dir / f"{model_name}/models/{model_name}"

    model = NucleiSegModel(model_path)

    try:
        with mlflow.start_run(run_id=os.environ.get("MLFLOW_RUN_ID")) as run:
            mlflow.pyfunc.log_model(
                artifact_path="mlflow_model",
                python_model=model,
                registered_model_name=os.environ.get("MLFLOW_MODEL_NAME"),
            )
            logger.info(
                f"Model registered successfully under the name: {os.environ.get('MLFLOW_MODEL_NAME')}"
            )
    except mlflow.exceptions.MlflowException as e:
        logger.info(f"Failed to register model: {e}")
    except Exception as e:
        logger.info(f"An unexpected error occurred: {e}")


def main():
    register_model()


if __name__ == "__main__":
    sys.exit(main())
