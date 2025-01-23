import os
import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import models  # Main model in models is not updated
import utilities

EXPE_NAME = "MH_Hyper_New_Hybrid"


def objective(trial):
    """
    Fonction objectif pour Optuna.
    """
    # Fixing a seed at each iteration to warrant reproducibility
    torch.manual_seed(21)
    np.random.seed(21)

    # set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(torch.cuda.get_device_name(device=None))

    # Initializing the model
    batch_size = trial.suggest_int("batch_size", 64, 128, step=64)
    output_weeks = 6
    # Hyperparameters
    num_epochs_entire = trial.suggest_int("num_epochs_entire", 10, 20, step=5)
    hidden_size = trial.suggest_int("hidden_size", 400, 750, step=50)
    num_lstm_layers = trial.suggest_int("num_lstm_layers", 2, 7)
    embedding_dims = [50, 50, 50, 50, 50, 50, 50, 570]
    num_fc_tabular_layers = trial.suggest_int("num_fc_tabular_layers", 1, 8)
    num_fc_combined_layers = trial.suggest_int("num_fc_combined_layers", 1, 8)
    dropout = trial.suggest_float("dropout", 0.1, 0.6, step=0.1)
    clip = trial.suggest_int("clip", 1, 5)
    # early stop parameters
    early_stop_patience = 5
    early_stop_min_delta = 0.001

    # Load the data
    dfs = utilities.load_dataFrames()
    train_loader = utilities.create_dataLoader(
        X_static=dfs["X_tabular_train"],
        X_static_cat=dfs["X_tabular_cat_train"],
        X_time=dfs["X_time_train"],
        y_target=dfs["y_target_train"],
        output_weeks=output_weeks,
        y_past=None,
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader = utilities.create_dataLoader(
        X_static=dfs["X_tabular_valid"],
        X_static_cat=dfs["X_tabular_cat_valid"],
        X_time=dfs["X_time_valid"],
        y_target=dfs["y_target_valid"],
        output_weeks=output_weeks,
        y_past=None,
        batch_size=batch_size,
        shuffle=False,
    )

    len_train_loader = len(
        train_loader
    )  # This line is necessary for the scheduler creation.
    class2id, id2class = utilities.setup_encoders_targets()
    model_name = f"{EXPE_NAME}_{trial.number}"
    model = models.New_HybridModel(
        num_numerical_features=dfs["X_tabular_train"].shape[-1],
        num_time_series_features=dfs["X_time_train"].shape[-1],
        hidden_size=hidden_size,
        list_unic_cat=dfs["list_cat"],
        embedding_dims=embedding_dims,
        num_lstm_layers=num_lstm_layers,
        num_fc_tabular_layers=num_fc_tabular_layers,
        num_fc_combined_layers=num_fc_combined_layers,
        output_size=output_weeks,
        dropout=dropout,
    )
    model.to(device)

    writer = SummaryWriter(f"{ROOT_TENSORBOARD}{model_name}/")
    valid_loss_min = np.inf

    early_stopping = utilities.EarlyStoppingObject(
        patience=early_stop_patience,
        min_delta=early_stop_min_delta,
        verbose=False,
    )
    counter = 0
    criterion = utilities.create_loss(trial)
    optimizer, scheduler = utilities.create_scheduler(
        trial,
        model_params=model.parameters(),
        len_train_loader=len_train_loader,
        num_epochs=num_epochs_entire,
    )

    for epoch in range(num_epochs_entire):
        for k, (inputs, static, static_cat, labels) in tqdm(enumerate(train_loader),
                                                            desc=f"Epoch {epoch+1}/{num_epochs_entire}",
                                                            total=len(train_loader),):
            model.train()
            counter += 1
            inputs, labels, static, static_cat = (
                inputs.to(device),
                labels.to(device),
                static.to(device),
                static_cat.to(device),
            )        
            optimizer.zero_grad()
            output = model(inputs, static, static_cat)
            loss = criterion(output, labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            scheduler.step()
            mse_loss = nn.MSELoss()(output, labels.float())

            with torch.no_grad():
                if k == len(train_loader) - 1 or k == (len(train_loader) - 1) // 2:
                    model.eval()
                    labels = []
                    raw_labels = []
                    preds = []
                    raw_preds = []
                    val_losses = []
                    val_mse_losses = []
                    for inp, stat, stat_cat, lab in valid_loader:
                        inp, lab, stat, stat_cat = inp.to(device), lab.to(device), stat.to(device), stat_cat.to(device)
                        out = model(inp, stat, stat_cat)
                        val_loss = criterion(out, lab.float())
                        val_losses.append(val_loss.item())
                        val_mse_loss = nn.MSELoss()(out, lab.float())
                        val_mse_losses.append(val_mse_loss.item())
                        for label in lab:
                            labels.append([int(l.round()) for l in label])
                            raw_labels.append([float(l) for l in label])
                        for pred in out:
                            preds.append([int(p.round()) for p in pred])
                            raw_preds.append([float(p) for p in pred])

                    labels = np.array(labels)
                    preds = np.clip(np.array(preds), 0, 5)
                    raw_labels = np.array(raw_labels)
                    raw_preds = np.array(raw_preds)

                    for i in range(output_weeks):
                        log_dict = {
                            "loss": float(loss),
                            "epoch": counter / len(train_loader),
                            "step": counter,
                            "lr": optimizer.param_groups[0]["lr"],
                            "week": i + 1,
                        }
                        # w=f'week_{i+1}_'
                        w = ""
                        log_dict[f"{w}validation_loss"] = np.mean(val_losses)
                        log_dict[f"{w}validation_mse_loss"] = np.mean(val_mse_losses)
                        log_dict[f"{w}macro_f1"] = f1_score(
                            labels[:, i], preds[:, i], average="macro"
                        )
                        log_dict[f"{w}micro_f1"] = f1_score(
                            labels[:, i], preds[:, i], average="micro"
                        )
                        log_dict[f"{w}mae"] = mean_absolute_error(
                            raw_labels[:, i], raw_preds[:, i]
                        )
                        print(log_dict)
                        writer.add_scalars(
                            "Loss",
                            {
                                "train": loss,
                                "validation": log_dict[f"{w}validation_loss"],
                            },
                            counter,
                        )
                        writer.add_scalars(
                            "Loss(MSE)",
                            {
                                "train": mse_loss,
                                "validation": log_dict[f"{w}validation_mse_loss"],
                            },
                            counter,
                        )
                        writer.add_scalars(
                            "F1(MSE)",
                            {
                                "macro": log_dict[f"{w}macro_f1"],
                                "micro": log_dict[f"{w}micro_f1"],
                            },
                            counter,
                        )
                        writer.add_scalar("MAE", log_dict[f"{w}mae"], counter)
                        writer.add_scalar("Learning-Rate", log_dict["lr"], counter)
                        for j, f1 in enumerate(
                            f1_score(labels[:, i], preds[:, i], average=None)
                        ):
                            log_dict[f"{w}{id2class[j]}_f1"] = f1

                    if np.mean(val_mse_losses) <= valid_loss_min:
                        torch.save(
                            model.state_dict(), f"{ROOT_MODELS_WEIGHTS}{model_name}.pt"
                        )
                        print(
                            "Validation MSE loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                                valid_loss_min, np.mean(val_mse_losses)
                            )
                        )
                        valid_loss_min = np.mean(val_mse_losses)

        early_stopping(valid_loss_min)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    return valid_loss_min


def optimize():
    """
    Fonction générique qui'enveloppe les expériences.
    """
    # Optuna study
    # Uncomment when re-running the optimization to delete the previous study
    optuna.delete_study(
        storage="sqlite:///optim_cible.sqlite3",
        study_name=EXPE_NAME,
    )
    study = optuna.create_study(
        storage="sqlite:///optim_cible.sqlite3",
        study_name=EXPE_NAME,
        direction="minimize",
    )
    study.optimize(objective, n_trials=50)  # Adjust the number of trials as needed
    print("\n")
    print("Best trial:")
    trial = study.best_trial
    print("    Number: {}".format(trial.number))
    print("    Value: {}".format(trial.value))

    print("    Params: ")
    for key, value in trial.params.items():
        print("       {}: {}".format(key, value))


if __name__ == "__main__":
    # Paths
    ROOT_RESULTS = f"results/{EXPE_NAME}/"
    ROOT_TENSORBOARD = f"runs/{EXPE_NAME}/"
    ROOT_MODELS_WEIGHTS = f"models/{EXPE_NAME}/"

    # Eliminate the previous results
    os.system(f"rm -rf {ROOT_RESULTS}")
    os.system(f"rm -rf {ROOT_TENSORBOARD}")
    os.system(f"rm -rf {ROOT_MODELS_WEIGHTS}")

    # Create the directories if they don't exist
    os.makedirs(ROOT_RESULTS, exist_ok=True)
    os.makedirs(ROOT_TENSORBOARD, exist_ok=True)
    os.makedirs(ROOT_MODELS_WEIGHTS, exist_ok=True)

    optimize()
