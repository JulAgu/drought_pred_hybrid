import os
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from torch.utils.tensorboard import SummaryWriter
import models as models # Main model in models is not updated
import utilities as utilities


def objective(trial):
    """
    Fonction objectif pour Optuna.
    """
    # Etablir le device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(torch.cuda.get_device_name(device=None))

    # Initialisation du modèle
    num_categorical_features = 7
    num_numerical_features = 22
    num_time_series_features = 21
    batch_size = 128
    output_weeks = 6
    # Hyperparameters
    num_epochs_entire = 10
    hidden_size = trial.suggest_int("hidden_size", 20, 600, step=20)
    num_lstm_layers = trial.suggest_int("num_lstm_layers", 1, 10)
    embedding_dims = trial.suggest_int("embedding_dims", 50, 300, step=10)
    num_fc_tabular_layers = trial.suggest_int("num_fc_tabular_layers", 1, 5)
    num_fc_combined_layers = trial.suggest_int("num_fc_combined_layers", 1, 5)
    dropout = trial.suggest_float("dropout", 0.1, 0.6, step=0.1)

    # Chargement des données cibles
    train_loader, val_loader, test_loader, list_cat = utilities.create_dataloaders(
        src_path=ROOT_SRC,
        results_path=ROOT_RESULTS,
        year=2023,
        mode="cible",
        wofost=True
    )

    # Modèle entier
    model_name = f"MLP_{trial.number}"
    model = models.onlyMLP(
        num_categorical_features,
        list_cat,
        num_numerical_features,
        embedding_dims,
        num_fc_tabular_layers,
        num_fc_combined_layers,
    ).to(device)
    criterion = utilities.create_loss(trial)
    optimizer = utilities.create_optimizer(trial, model.parameters())
    # TensorBoard
    writer = SummaryWriter(ROOT_TENSORBOARD + model_name)
    # Entrainement utilities
    checkpoint_path = ROOT_MODELS_WEIGHTS + f"{model_name}_checkpoint.pth"
    best_model_path = ROOT_MODELS_WEIGHTS + f"{model_name}_best.pth"
    # Entrainement
    utilities.train_model(
        model_name=model_name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs_entire,
        checkpoint_path=checkpoint_path,
        best_model_path=best_model_path,
        writer=writer,
        early_stop_patience=10,
        early_stop_min_delta=0.001, # Change this
        device_cuda=device,
        kind="mlp",
    )
    # Evaluation du modèle
    model.load_state_dict(torch.load(best_model_path))
    utilities.report_and_basic_graph(
        model_name=model_name,
        model=model,
        test_loader=test_loader,
        kind_of_model="mlp",
        results_path=ROOT_RESULTS,
        device_cuda=device,
    )

    full_val_loss = utilities.loss_for_hypertuning(model, val_loader, "mlp", device)

    return full_val_loss

if __name__ == "__main__":
    # Fixing a seed to warrant the reproducibility
    torch.manual_seed(21)

    NAME_DIRECTORY = "MH_Hyper"
    # Paths for the remote server
    ROOT_SRC = "src/processed/"
    ROOT_RESULTS = f"results/{NAME_DIRECTORY}/"
    ROOT_TENSORBOARD = f"runs/{NAME_DIRECTORY}/"
    ROOT_MODELS_WEIGHTS = f"models/{NAME_DIRECTORY}/"