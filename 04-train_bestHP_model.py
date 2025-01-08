import os
import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import models
import utilities

EXPE_NAME = "MH_Hybrid_manual"

def one_train(ablation_tabular=False,
                   ablation_TS=False,
                   ablation_attention=False,
                   etiquette="",):
    # set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(torch.cuda.get_device_name(device=None))

    # Initializing the model
    num_categorical_features = 7
    num_numerical_features = 22
    num_time_series_features = 21
    batch_size = 128
    output_weeks = 6
    # Hyperparameters
    num_epochs_entire = 20
    hidden_size = 340
    num_lstm_layers = 8
    embedding_dims = 270
    num_fc_tabular_layers = 4 
    num_fc_combined_layers = 1
    dropout = 0.30000000000000004
    # early stop parameters
    early_stop_patience = 10
    early_stop_min_delta = 0.001
    lr = 0.0005

    # Load the data
    dfs = utilities.load_dataFrames()
    train_loader = utilities.create_dataLoader(
        X_static=dfs["X_tabular_train"],
        X_static_cat=dfs["X_tabular_cat_train"],
        X_time=dfs["X_time_train"],
        y_target=dfs["y_target_train"],
        output_weeks=output_weeks,
        y_past=None,
        batch_size=128,
        shuffle=True,
    )
    valid_loader = utilities.create_dataLoader(
        X_static=dfs["X_tabular_valid"],
        X_static_cat=dfs["X_tabular_cat_valid"],
        X_time=dfs["X_time_valid"],
        y_target=dfs["y_target_valid"],
        output_weeks=output_weeks,
        y_past=None,
        batch_size=128,
        shuffle=False,
    )

    len_train_loader = len(
        train_loader
    )  # This line is necessary for the scheduler creation.
    class2id, id2class = utilities.setup_encoders_targets()
    model_name = f"{EXPE_NAME}_{etiquette}"
    model = models.HybridModel(
        num_categorical_features=num_categorical_features,
        list_unic_cat=dfs["list_cat"],
        num_numerical_features=num_numerical_features,
        num_time_series_features=num_time_series_features,
        hidden_size=hidden_size,
        num_lstm_layers=num_lstm_layers,
        dropout=dropout,
        embedding_dims=embedding_dims,
        num_fc_tabular_layers=num_fc_tabular_layers,
        num_fc_combined_layers=num_fc_combined_layers,
        output_size=output_weeks,
        ablation_TS=ablation_TS,
        ablation_tabular=ablation_tabular,
        ablation_attention=ablation_attention,
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
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=lr,
                                                    steps_per_epoch=len_train_loader,
                                                    epochs=num_epochs_entire,
                                                    )

    for epoch in range(num_epochs_entire):
        for k, batch in tqdm(
            enumerate(train_loader),
            desc=f"Epoch {epoch+1}/{num_epochs_entire}",
            total=len(train_loader),
        ):
            X_static, X_static_cat, X_time, y_target = [
                data.to(device) for data in batch
            ]
            model.train()
            counter += 1
            optimizer.zero_grad()
            output = model(X_static_cat, X_static, X_time)

            loss = criterion(output, y_target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                if k == len(train_loader) - 1 or k == (len(train_loader) - 1) // 2:
                    model.eval()
                    labels = []
                    raw_labels = []
                    preds = []
                    raw_preds = []
                    val_losses = []
                    for batch in valid_loader:
                        X_static_val, X_static_cat_val, X_time_val, y_target_val = [
                            data.to(device) for data in batch
                        ]
                        output = model(X_static_cat_val, X_static_val, X_time_val)
                        val_loss = criterion(output, y_target_val)
                        val_losses.append(val_loss.item())
                        for label in y_target_val:
                            labels.append([int(l.round()) for l in label])
                            raw_labels.append([float(l) for l in label])
                        for pred in output:
                            preds.append([int(p.round()) for p in pred])
                            raw_preds.append([float(p) for p in pred])

                    labels = np.array(labels)
                    preds = np.clip(np.array(preds), 0, 5)
                    raw_labels = np.array(raw_labels)
                    raw_preds = np.array(raw_preds)
                    f1_macro = f1_score(
                        labels.flatten(), preds.flatten(), average="macro"
                    )

                    for i in range(output_weeks):
                        log_dict = {
                            "loss": float(loss),
                            "epoch": counter / len(train_loader),
                            "step": counter,
                            "lr": optimizer.param_groups[0]["lr"],
                            "week": i + 1,
                        }
                        # w = f'week_{i+1}_'
                        w = ""
                        log_dict[f"{w}validation_loss"] = np.mean(val_losses)
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
                            "Loss(custom)",
                            {
                                "train": loss,
                                "validation": log_dict[f"{w}validation_loss"],
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
                        model.train()
                    if np.mean(val_losses) <= valid_loss_min:
                        torch.save(
                            model.state_dict(), f"{ROOT_MODELS_WEIGHTS}{model_name}.pt"
                        )
                        print(
                            "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                                valid_loss_min, np.mean(val_losses)
                            )
                        )
                        valid_loss_min = np.mean(val_losses)

        early_stopping(valid_loss_min)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    return valid_loss_min

if __name__ == "__main__":
    # Fixing a seed to warrant the reproducibility
    torch.manual_seed(21)
    np.random.seed(21)

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

    no_tab_no_att = one_train(etiquette="just_a_test") 