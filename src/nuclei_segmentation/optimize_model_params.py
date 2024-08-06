from dotenv import load_dotenv
import numpy as np
from pathlib import Path
import os, sys
import shutil
import yaml

from cellpose import core, utils, io, models, metrics
import wandb
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
from sklearn.model_selection import KFold


load_dotenv('.env') # loading wandb credentials


def read_exp_config():
    code_path = Path(__file__).parent
    config_path = code_path / 'configs/optuna_experiments_config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(config)

    return config


CONFIG = read_exp_config()

WANDB_API_KEY = os.environ['wandb_api_key']
OPTUNA_TRIALS = CONFIG['optuna_trials']

traet_str = 'treated'
if CONFIG['not_treated_exp']:
    traet_str = 'not_treated'

if CONFIG['on_test']:
    project_name = f'optuna_fine_tuned_cell_mask_constructor_on_test_{traet_str}'
else:
    if CONFIG['few_shot']:
        project_name = f'optuna_fine_tuned_cell_mask_constructor_{traet_str}_few_shot'
    else:
        project_name = f'optuna_fine_tuned_cell_mask_constructor_{traet_str}'
wandbc = WeightsAndBiasesCallback(wandb_kwargs={'project': project_name}, as_multirun=True)


@wandbc.track_in_wandb()
def objective(trial):
    root_path = Path(__file__).parent.parent.parent.resolve()

    data_path = root_path / 'data' / 'nuclei_dataset' / 'train_by_exps'

    train_dirs_list = []
    if CONFIG['not_treated_exp']:
        train_dirs_list.append(os.path.join(data_path, 'Nthyori,exp3,cond3'))
        train_dirs_list.append(os.path.join(data_path, 'Nthyori,exp5,cond1a,control'))
    else:  
        train_dirs_list.append(os.path.join(data_path, 'Nthyori,exp3,cond3'))
        train_dirs_list.append(os.path.join(data_path, 'Nthyori,exp5,cond1a,control'))
        train_dirs_list.append(os.path.join(data_path, 'Nthyori,exp4,cond3'))
    test_dir = os.path.join(data_path, 'test')

    model_type = 'cyto2'
    chan = 3
    chan2 = 0
    channels = [chan, chan2]

    params = {}
    params['n_epochs'] = trial.suggest_categorical('n_epochs', [20, 30, 50, 70, 100])
    params['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 5e-1, log=True)
    params['weight_decay'] = trial.suggest_float('weight_decay', 1e-5, 5e-2, log=True)
    params['nimg_per_epoch'] = trial.suggest_categorical('nimg_per_epoch', [5, 10, 20, 30, 40, 45, 50])
    params['batch_size'] = trial.suggest_categorical('batch_size', [1, 4, 8, 16, 32])

    # model_name = 'fine_tuned_cell_mask_constructor'
    model_name = model_name = project_name

    # Workflow
    # out_dir = f'/output/{project_name}'

    # CLI
    out_dir = root_path / 'models'

    model_path = os.path.join(out_dir, model_name)

    train_data_list = []
    train_labels_list = []

    for train_dir in train_dirs_list:
        output = io.load_train_test_data(train_dir, None, mask_filter='_seg.npy')
        train_data, train_labels, _, _, _, _ = output
        train_data_list.append(train_data)
        train_labels_list.append(train_labels)

    kf = KFold(n_splits=5)
    kf_iters_list = []
    for i in range(len(train_dirs_list)):
        kf_iters_list.append(iter(kf.split(train_data_list[i])))

    ap_kfolds = []
    for _ in range(kf.n_splits):
        train_data_fold, train_labels_fold, val_data_fold, val_labels_fold = [], [], [], []
        for i in range(len(kf_iters_list)):
            train_inds, val_inds = next(kf_iters_list[i])
            train_data_fold.append(np.array(train_data_list[i])[train_inds])
            train_labels_fold.append(np.array(train_labels_list[i])[train_inds])
            val_data_fold.append(np.array(train_data_list[i])[val_inds])
            val_labels_fold.append(np.array(train_labels_list[i])[val_inds])

        train_data_fold = np.concatenate(train_data_fold)
        train_labels_fold = np.concatenate(train_labels_fold)
        val_data_fold = np.concatenate(val_data_fold)
        val_labels_fold = np.concatenate(val_labels_fold)

        model = models.CellposeModel(gpu=True, model_type=model_type)

        trained_model_path = model.train(list(train_data_fold), list(train_labels_fold), 
                                        test_data=list(val_data_fold),
                                        test_labels=list(val_labels_fold),
                                        channels=channels, 
                                        save_path=model_path, 
                                        n_epochs=params['n_epochs'],
                                        learning_rate=params['learning_rate'], 
                                        weight_decay=params['weight_decay'], 
                                        nimg_per_epoch=params['nimg_per_epoch'],
                                        model_name=model_name)

        model = models.CellposeModel(gpu=True, pretrained_model=trained_model_path)
        diameter = 0
        diameter = model.diam_labels if diameter==0 else diameter

        masks = model.eval(list(val_data_fold), 
                        channels=[chan, chan2],
                        diameter=diameter)[0]

        ap = metrics.average_precision(list(val_labels_fold), masks)[0]
        ap_8 = ap[:, 0].mean()
        ap_kfolds.append(ap_8)

    return np.mean(ap_kfolds)


@wandbc.track_in_wandb()
def objective_on_test(trial):
    root_path = Path(__file__).parent.parent.parent.resolve()

    data_path = root_path / 'data' / 'nuclei_dataset'

    train_dirs_list = []
    if CONFIG['not_treated_exp']:
        train_dir = os.path.join(data_path, 'train_set_not_treated')
    else:
        train_dir = os.path.join(data_path, 'train_set')
    test_dir = os.path.join(data_path, 'test')

    p = Path(train_dir)
    train_dataset_len = len(list(p.rglob("*_seg.npy")))

    model_type = 'cyto2'
    chan = 3
    chan2 = 0
    channels = [chan, chan2]

    params = {}
    params['n_epochs'] = trial.suggest_categorical('n_epochs', [20, 30, 50, 70, 100])
    params['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 5e-1, log=True)
    params['weight_decay'] = trial.suggest_float('weight_decay', 1e-5, 5e-2, log=True)
    params['nimg_per_epoch'] = trial.suggest_categorical('nimg_per_epoch', [train_dataset_len, int(train_dataset_len * 1.5), int(train_dataset_len * 2), int(train_dataset_len * 3)])
    params['batch_size'] = trial.suggest_categorical('batch_size', [1, 4, 8, 16, 32])

    # model_name = 'fine_tuned_cell_mask_constructor'
    model_name = project_name

    # Workflow
    out_dir = f'/output/{project_name}'

    # CLI
    # out_dir = root_path / 'models'

    model_path = os.path.join(out_dir, model_name)

    output = io.load_train_test_data(train_dir, test_dir, mask_filter='_seg.npy')
    train_data, train_labels, _, test_data, test_labels, _ = output

    model = models.CellposeModel(gpu=True, model_type=model_type)

    trained_model_path = model.train(train_data, train_labels, 
                                    test_data=test_data,
                                    test_labels=test_labels,
                                    channels=channels, 
                                    save_path=model_path, 
                                    n_epochs=params['n_epochs'],
                                    learning_rate=params['learning_rate'], 
                                    weight_decay=params['weight_decay'], 
                                    nimg_per_epoch=params['nimg_per_epoch'],
                                    model_name=model_name)

    model = models.CellposeModel(gpu=True, pretrained_model=trained_model_path)
    diameter = 0
    diameter = model.diam_labels if diameter==0 else diameter

    output = io.load_train_test_data(test_dir, mask_filter='_seg.npy')
    test_data, test_labels = output[:2]

    masks = model.eval(test_data, 
                    channels=[chan, chan2],
                    diameter=diameter)[0]

    ap = metrics.average_precision(test_labels, masks)[0]
    ap_8 = ap[:, 0].mean()

    return ap_8


@wandbc.track_in_wandb()
def objective_few_shot(trial):
    root_path = Path(__file__).parent.parent.parent.resolve()

    data_path = root_path / 'data' / 'nuclei_dataset' / 'train_by_exps'

    train_dirs_list = []
    if CONFIG['not_treated_exp']:
        train_dirs_list.append(os.path.join(data_path, 'Nthyori,exp3,cond3'))
        train_dirs_list.append(os.path.join(data_path, 'Nthyori,exp5,cond1a,control'))
    else:  
        train_dirs_list.append(os.path.join(data_path, 'Nthyori,exp3,cond3'))
        train_dirs_list.append(os.path.join(data_path, 'Nthyori,exp5,cond1a,control'))
        train_dirs_list.append(os.path.join(data_path, 'Nthyori,exp4,cond3'))
    test_dir = os.path.join(data_path, 'test')

    model_type = 'cyto2'
    chan = 3
    chan2 = 0
    channels = [chan, chan2]

    params = {}
    params['n_epochs'] = trial.suggest_categorical('n_epochs', [20, 30, 50, 70])
    params['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 5e-1, log=True)
    params['weight_decay'] = trial.suggest_float('weight_decay', 1e-5, 5e-2, log=True)
    params['nimg_per_epoch'] = trial.suggest_categorical('nimg_per_epoch', [5, 10, 20, 30, 35])
    params['batch_size'] = trial.suggest_categorical('batch_size', [1, 4, 8, 16])
    params['n_support'] = trial.suggest_int('n_support', 1, 5, log=True)

    # model_name = 'fine_tuned_cell_mask_constructor'
    model_name = project_name

    # Workflow
    out_dir = f'/output/{project_name}'

    # CLI
    # out_dir = root_path / 'models'

    model_path = os.path.join(out_dir, model_name)

    train_data_list = []
    train_labels_list = []

    for train_dir in train_dirs_list:
        output = io.load_train_test_data(train_dir, None, mask_filter='_seg.npy')
        train_data, train_labels, _, _, _, _ = output
        train_data_list.append(train_data)
        train_labels_list.append(train_labels)

    kf = KFold(n_splits=3)
    kf_iters_list = []
    for i in range(len(train_dirs_list)):
        kf_iters_list.append(iter(kf.split(train_data_list[i])))

    ap_kfolds = []
    for _ in range(kf.n_splits):
        val_data_fold, val_labels_fold = [], []
        train_inds_list = []
        for i in range(len(kf_iters_list)):
            train_inds, val_inds = next(kf_iters_list[i])
            train_inds_list.append(train_inds)
            val_data_fold.append(np.array(train_data_list[i])[val_inds])
            val_labels_fold.append(np.array(train_labels_list[i])[val_inds])

        val_data_fold = np.concatenate(val_data_fold)
        val_labels_fold = np.concatenate(val_labels_fold)

        for __ in range(3):
            train_data_fold, train_labels_fold = [], []
            for j in range(len(train_inds_list)): 
                support_ind = np.random.choice(train_inds_list[j], params['n_support'], replace=False)
                train_data_fold.append(np.array(train_data_list[j])[support_ind])
                train_labels_fold.append(np.array(train_labels_list[j])[support_ind])

            train_data_fold = np.concatenate(train_data_fold)
            train_labels_fold = np.concatenate(train_labels_fold)

            model = models.CellposeModel(gpu=True, model_type=model_type)

            trained_model_path = model.train(list(train_data_fold), list(train_labels_fold), 
                                            test_data=list(val_data_fold),
                                            test_labels=list(val_labels_fold),
                                            channels=channels, 
                                            save_path=model_path, 
                                            n_epochs=params['n_epochs'],
                                            learning_rate=params['learning_rate'], 
                                            weight_decay=params['weight_decay'], 
                                            nimg_per_epoch=params['nimg_per_epoch'],
                                            model_name=model_name)

            model = models.CellposeModel(gpu=True, pretrained_model=trained_model_path)
            diameter = 0
            diameter = model.diam_labels if diameter==0 else diameter

            masks = model.eval(list(val_data_fold), 
                            channels=[chan, chan2],
                            diameter=diameter)[0]

            ap = metrics.average_precision(list(val_labels_fold), masks)[0]
            ap_8 = ap[:, 0].mean()
            ap_kfolds.append(ap_8)

    return np.mean(ap_kfolds)


def optimize_params_optuna():
    if CONFIG['on_test']:
        objective_function = objective_on_test
    else:
        if CONFIG['few_shot']:
            objective_function = objective_few_shot
        else:
            objective_function = objective

    wandb.login(key=WANDB_API_KEY)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective_function, n_trials=OPTUNA_TRIALS, callbacks=[wandbc])


def main():
    optimize_params_optuna()


if __name__ == '__main__':
    sys.exit(main())
 

