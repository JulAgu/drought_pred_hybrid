import os
import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import optuna
import pickle
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import models
import utilities
# import datasets without the categories

X_tabular_train = np.load("data/processed_no_cat_and_no_tensors/X_tabular_train.npy")
X_time_train = np.load("data/processed_no_cat_and_no_tensors/X_time_train.npy")
y_target_train = np.load("data/processed_no_cat_and_no_tensors/y_target_train.npy")
X_tabular_validation = np.load("data/processed_no_cat_and_no_tensors/X_tabular_validation.npy")
X_time_valid = np.load("data/processed_no_cat_and_no_tensors/X_time_valid.npy")
y_target_valid = np.load("data/processed_no_cat_and_no_tensors/y_target_valid.npy")
with open(f"data/processed_no_cat_and_no_tensors/valid_fips.pickle", "rb") as f:
    valid_fips = pickle.load(f)

EXPE_NAME = "MH_Hybrid_2Outputs"

def twoLoss_train(ablation_tabular=False,
                  ablation_TS=False,
                  ablation_attention=False,
                  etiquette="",):

    # Fixing a seed to warrant the reproducibility
    torch.manual_seed(21)
    np.random.seed(21)

    # set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(torch.cuda.get_device_name(device=None))

    # Initializing the model
    num_numerical_features = 0 # Not used
    num_time_series_features = 21
    batch_size = 128
    output_weeks = 6
    # Hyperparameters
    num_epochs_entire = 15
    hidden_size = 100 #340
    num_lstm_layers = 5 #6
    num_fc_tabular_layers = 3
    num_fc_combined_layers = 1
    dropout = 0.4
    # early stop parameters
    early_stop_patience = 6
    early_stop_min_delta = 0.001
    lr = 0.0008

    # Load the data
    y_train_tensor = torch.tensor(y_target_train[:, :output_weeks])
    # Flatten the target tensor to compute class frequencies
    flat_labels = y_train_tensor.view(-1).numpy().round().astype(int)  # Shape: [observations * weeks]
    class_counts = np.bincount(flat_labels)  # Count occurrences of each class
    class_weights = 1.0 / class_counts       # Inverse of class frequencies
    # Map class weights to each [i, j] in the target tensor
    weights_per_sample = class_weights[y_train_tensor.numpy().round().astype(int)]  # Shape: [observations, weeks]
    # Compute the average weight per observation (averaging over the 6 weeks)
    sample_weights = weights_per_sample.mean(axis=1)  # Shape: [observations]
    # Create a sampler for the training data
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # Prepare the datasets
    train_data = TensorDataset(
        torch.tensor(X_time_train).type(torch.FloatTensor),
        torch.tensor(X_tabular_train).type(torch.FloatTensor),
        y_train_tensor.type(torch.FloatTensor),
    )
    valid_data = TensorDataset(
        torch.tensor(X_time_valid).type(torch.FloatTensor),
        torch.tensor(X_tabular_validation).type(torch.FloatTensor),
        torch.tensor(y_target_valid[:, :output_weeks]).type(torch.FloatTensor),
    )

    # DataLoaders with sampler for training and default for validation
    train_loader = DataLoader(
        train_data, batch_size=batch_size, drop_last=False
    )

    valid_loader = DataLoader(
        valid_data, shuffle=False, batch_size=batch_size, drop_last=False
    )
    len_train_loader = len(
        train_loader
    )  # This line is necessary for the scheduler creation.
    class2id, id2class = utilities.setup_encoders_targets()
    model_name = f"{EXPE_NAME}_{etiquette}"
    model = models.HybridModel_2Outputs_noEmbbedings(
        num_numerical_features=X_tabular_train.shape[-1],
        num_time_series_features=X_time_train.shape[-1],
        hidden_size=hidden_size,
        num_lstm_layers=num_lstm_layers,
        dropout=dropout,
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
    criterion = nn.MSELoss()
    criterion2 = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
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
            X_time, X_static, y_target = [
                data.to(device) for data in batch
            ]
            model.train()
            counter += 1
            optimizer.zero_grad()
            output, output_cat = model(X_time, X_static)

            loss1 = criterion(output, y_target)
            loss2 = criterion2(output_cat, y_target.round().long())
            loss = loss1 + loss2
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
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
                    val_losses2 = []
                    val_losses_combined = []
                    for batch in valid_loader:
                        X_time_val, X_static_val, y_target_val = [
                            data.to(device) for data in batch
                        ]
                        output, output_cat = model(X_time_val, X_static_val)
                        # val_loss = criterion(output, y_target_val)
                        val_loss = 0 # This is new
                        val_loss2 = criterion2(output_cat, y_target_val.round().long())
                        # val_losses.append(val_loss.item())
                        val_losses.append(val_loss) # This is new
                        val_losses2.append(val_loss2.item())
                        # val_losses_combined.append(val_loss.item() + val_loss2.item()) 
                        val_losses_combined.append(val_loss + val_loss2.item())# This is new
                        
                        for label in y_target_val:
                            labels.append([int(l.round()) for l in label])
                            raw_labels.append([float(l) for l in label])
                        for pred, pred_cat in zip(output, output_cat):
                            raw_preds.append([float(p) for p in pred])
                            preds.append(torch.max(pred_cat, 1)[1].cpu().numpy()) #[1] because the return of torch.max is a tuple where [0] is the max value and [1] is the index of the max value.                          
                    print(pred_cat)
                    print(torch.max(pred_cat, 1))
                    labels = np.array(labels)
                    preds = np.array(preds) # I changed this !
                    raw_labels = np.array(raw_labels)
                    raw_preds = np.array(raw_preds)
                    print("labels_shape\n")
                    print(labels.shape)
                    print("preds_shape\n")
                    print(preds.shape)
                    print("labels_first_element\n")
                    print(labels[0])
                    print("preds_first_element\n")
                    print(preds[0])
                    for i in range(output_weeks):
                        log_dict = {
                            "loss": float(loss),
                            "Hubberloss": float(loss1),
                            "CrossEntropyLoss": float(loss2),
                            "epoch": counter / len(train_loader),
                            "step": counter,
                            "lr": optimizer.param_groups[0]["lr"],
                            "week": i + 1,
                        }
                        # w = f'week_{i+1}_'
                        w = ""
                        log_dict[f"{w}validation_Hubberloss"] = np.mean(val_losses)
                        log_dict[f"{w}validation_crossEntropyLoss"] = np.mean(val_losses2)
                        log_dict[f"{w}validation_combined_loss"] = np.mean(val_losses_combined)
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
                                "validation": log_dict[f"{w}validation_combined_loss"],
                            },
                            counter,
                        )
                        writer.add_scalars(
                            "HubberLoss",
                            {
                                "train": loss1,
                                "validation": log_dict[f"{w}validation_Hubberloss"],
                            },
                            counter,
                        )
                        writer.add_scalars(
                            "CrossEntropyLoss",
                            {
                                "train": loss2,
                                "validation": log_dict[f"{w}validation_crossEntropyLoss"],
                            },
                            counter,
                        )
                        writer.add_scalars(
                            "F1",
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

                    if np.mean(val_losses_combined) <= valid_loss_min:
                        torch.save(
                            model.state_dict(), f"{ROOT_MODELS_WEIGHTS}{model_name}.pt"
                        )
                        print(
                            "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                                valid_loss_min, np.mean(val_losses_combined)
                            )
                        )
                        valid_loss_min = np.mean(val_losses_combined)

        early_stopping(valid_loss_min)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    return valid_loss_min

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

    twoLoss = twoLoss_train(etiquette="twoLoss")

    print(f"twoLoss: {twoLoss}")
