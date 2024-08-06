from dotenv import load_dotenv
from pathlib import Path
import os, sys
import mlflow

from cellpose import models


load_dotenv('.env') # loading mlflow credentials

class NucleiSegModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model_path):
        super().__init__()
        self.model = models.CellposeModel(gpu=False, pretrained_model=model_path)

def register_model(run_id_to_register, registered_model_name):
    mlflow.set_experiment(registered_model_name)

    root_path = Path(__file__).parent.parent.parent.resolve()
    out_dir = root_path / 'models'
    model_name = 'cellpose_thyroid'
    model_path = os.path.join(out_dir, model_name, 'models', model_name)

    model = NucleiSegModel(model_path)

    try:
        with mlflow.start_run(run_id=run_id_to_register) as run:
            mlflow.pyfunc.log_model(
                artifact_path='mlflow_model',
                python_model=model,
                registered_model_name=registered_model_name,
            )
            print(f"Model registered successfully under the name: {registered_model_name}")
    except mlflow.exceptions.MlflowException as e:
        print(f"Failed to register model: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    run_id_to_register = '287203b15318462d82f7bc333410f953'
    registered_model_name = 'va_nuclei_segmentation'
    register_model(run_id_to_register, registered_model_name)


if __name__ == '__main__':
    sys.exit(main())