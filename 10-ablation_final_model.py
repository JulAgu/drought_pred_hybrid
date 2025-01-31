import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from scipy.interpolate import interp1d
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, mean_absolute_error, root_mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter


is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("using GPU")
else:
    device = torch.device("cpu")
    print("using CPU")


EXPE_NAME = "Ablation_final"

class DroughtNet(nn.Module):
    def __init__(
        self,
        output_size,
        num_input_features,
        hidden_dim,
        n_layers,
        ffnn_layers,
        drop_prob,
        static_dim,
        list_unic_cat,
        embedding_dims,
        embeddings_dropout,
        ablation_TS=False,
        ablation_tabular=False,
        ablation_attention=False,
    ):
        super(DroughtNet, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.ablation_TS = ablation_TS
        self.ablation_tabular = ablation_tabular
        self.ablation_attention = ablation_attention

        if not self.ablation_tabular:
            self.embeddings = nn.ModuleList(
                    [
                        nn.Embedding(num_embeddings=i, embedding_dim=dimension)
                        for i, dimension in zip(list_unic_cat, embedding_dims)
                    ]
                )
            self.embeddings_dropout = nn.Dropout(embeddings_dropout)
            self.after_embeddings = nn.Sequential(nn.Linear(sum(embedding_dims), 7), nn.ReLU())

        self.lstm = nn.LSTM(
            num_input_features,
            hidden_dim,
            n_layers,
            dropout=drop_prob,
            batch_first=True,
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(drop_prob)
        self.fflayers = []
        
        input_size = hidden_dim * 2 + static_dim + 7
        intermediate_size = hidden_dim
        exit_size = hidden_dim

        if self.ablation_TS:
            input_size = static_dim + 7
            intermediate_size = static_dim + 7
            exit_size = static_dim + 7

        if self.ablation_tabular:
            input_size = hidden_dim * 2
            intermediate_size = hidden_dim
            exit_size = hidden_dim

        if self.ablation_attention and self.ablation_tabular:
            input_size = hidden_dim
            intermediate_size = hidden_dim
            exit_size = hidden_dim

        elif self.ablation_attention and not self.ablation_tabular:
            input_size = hidden_dim + static_dim + 7
            intermediate_size = hidden_dim + static_dim + 7
            exit_size = hidden_dim + static_dim + 7         

        for i in range(ffnn_layers - 1):
            if i == 0:
                self.fflayers.append(nn.Linear(input_size, intermediate_size))
            else:
                self.fflayers.append(nn.Linear(intermediate_size, exit_size))
        self.fflayers = nn.ModuleList(self.fflayers)
        self.final = nn.Linear(exit_size, output_size)

    def forward(self, x, hidden, static, cat):
        batch_size = x.size(0)
        x = x.to(dtype=torch.float32)
        static = static.to(dtype=torch.float32)

        if not self.ablation_TS:
            if self.ablation_attention:
                lstm_out, hidden = self.lstm(x, hidden)
                last_hidden = lstm_out[:, -1, :]
                out = self.dropout(last_hidden)
            else:
                lstm_out, hidden = self.lstm(x, hidden)
                last_hidden = lstm_out[:, -1, :]
                attn_weights = F.softmax(self.attention(lstm_out), dim=1)
                context_vector = torch.sum(attn_weights * lstm_out, dim=1)
                att_out = torch.cat((context_vector, last_hidden), 1)
                out = self.dropout(att_out)

        if not self.ablation_tabular:
            embeddings = [emb(cat[:, i]) for i, emb in enumerate(self.embeddings)]
            cat = torch.cat(embeddings, dim=1)
            cat = self.embeddings_dropout(cat)
            cat = self.after_embeddings(cat)

        if self.ablation_TS:
            for i in range(len(self.fflayers)):
                if i == 0 and static is not None:
                    out = self.fflayers[i](torch.cat((static, cat), 1))
                else:
                    out = self.fflayers[i](out)
            out = self.final(out)

        elif self.ablation_tabular:
            for i in range(len(self.fflayers)):
                out = self.fflayers[i](out)
            out = self.final(out)

        else:
            for i in range(len(self.fflayers)):
                if i == 0 and static is not None:
                    out = self.fflayers[i](torch.cat((out, static, cat), 1))
                else:
                    out = self.fflayers[i](out)
            out = self.final(out)

        out = out.view(batch_size, -1)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
        )
        return hidden


def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

scaler_dict = {}
scaler_dict_static = {}
scaler_dict_past = {}
def normalize(X_static, X_time, y_past=None, fit=False):
    for index in tqdm(range(X_time.shape[-1])):
        if fit:
            scaler_dict[index] = RobustScaler().fit(X_time[:, :, index].reshape(-1, 1))
        X_time[:, :, index] = (
            scaler_dict[index]
            .transform(X_time[:, :, index].reshape(-1, 1))
            .reshape(-1, X_time.shape[-2])
        )
    for index in tqdm(range(X_static.shape[-1])):
        if fit:
            scaler_dict_static[index] = RobustScaler().fit(
                X_static[:, index].reshape(-1, 1)
            )
        X_static[:, index] = (
            scaler_dict_static[index]
            .transform(X_static[:, index].reshape(-1, 1))
            .reshape(1, -1)
        )
    index = 0
    if y_past is not None:
        if fit:
            scaler_dict_past[index] = RobustScaler().fit(y_past.reshape(-1, 1))
        y_past[:, :] = (
            scaler_dict_past[index]
            .transform(y_past.reshape(-1, 1))
            .reshape(-1, y_past.shape[-1])
        )
        return X_static, X_time, y_past
    return X_static, X_time

dataDic = {"train": pd.read_csv("src/train_timeseries/train_timeseries.csv"),
           "test": pd.read_csv("src/test_timeseries/test_timeseries.csv"),
           "validation": pd.read_csv("src/validation_timeseries/validation_timeseries.csv"),
           "soil" : pd.read_csv("src/soil_data.csv"),
           }

class2id = {
    'None': 0,
    'D0': 1,
    'D1': 2,
    'D2': 3,
    'D3': 4,
    'D4': 5,
}
id2class = {v: k for k, v in class2id.items()}

dfs = {
    k: dataDic[k].set_index(['fips', 'date'])
    for k in dataDic.keys() if k != "soil"
}
dfs["soil"] = dataDic["soil"]

# import
import pickle
with open("data/data.pkl", "rb") as f:
    data = pickle.load(f)
    X_tabular_train = data["X_tabular_train"]
    X_time_train = data["X_time_train"]
    y_target_train = data["y_target_train"]
    X_tabular_validation = data["X_tabular_validation"]
    X_time_valid = data["X_time_valid"]
    y_target_valid = data["y_target_valid"]
    valid_fips = data["valid_fips"]
    X_tabular_test = data["X_tabular_test"]
    X_time_test = data["X_time_test"]
    y_target_test = data["y_target_test"]
    test_fips = data["test_fips"]

ordered_cols = sorted([c for c in dfs["soil"].columns if c not in ["soil", "lat", "lon"]])
cat_cols = [ordered_cols.index(i) for i in ["SQ1", "SQ2", "SQ3", "SQ4", "SQ5", "SQ6", "SQ7"]]

X_tabular_cat_train = X_tabular_train[:,cat_cols].astype(int)
X_tabular_train = X_tabular_train[:,[i for i in range(X_tabular_train.shape[1]) if i not in cat_cols]]
X_tabular_cat_valid = X_tabular_validation[:,cat_cols].astype(int)
X_tabular_validation = X_tabular_validation[:,[i for i in range(X_tabular_validation.shape[1]) if i not in cat_cols]]
X_tabular_cat_test = X_tabular_test[:,cat_cols].astype(int)
X_tabular_test = X_tabular_test[:,[i for i in range(X_tabular_test.shape[1]) if i not in cat_cols]]

dico_trad = {}
for cat in range(X_tabular_cat_train.shape[1]):
    dico_trad[cat] = {j: i for i,j in enumerate(sorted(np.unique_values(X_tabular_cat_train[:,cat])))}
    dico_trad[cat]["unknown"] = len(np.unique_values(X_tabular_cat_train[:,cat]))

for cat in range(len(cat_cols)):
    X_tabular_cat_train[:,cat] = [dico_trad[cat][i] for i in X_tabular_cat_train[:,cat]]
    X_tabular_cat_valid[:,cat] = [dico_trad[cat][i] if i in dico_trad[cat] else dico_trad[cat]["unknown"] for i in X_tabular_cat_valid[:,cat]]
    X_tabular_cat_test[:,cat] = [dico_trad[cat][i] if i in dico_trad[cat] else dico_trad[cat]["unknown"] for i in X_tabular_cat_test[:,cat]]

X_tabular_train, X_time_train = normalize(X_tabular_train, X_time_train, fit=True)
X_tabular_validation, X_time_valid = normalize(X_tabular_validation, X_time_valid)
X_tabular_test, X_time_test = normalize(X_tabular_test, X_time_test)

######  Fixed Hyperparameters ###### 
batch_size = 128
output_weeks = 6
hidden_dim = 490
n_layers = 2
ffnn_layers = 2
dropout = 0.1
lr = 7e-5
epochs = 9
clip = 5
embed_dim = [3, 3, 3, 3, 3, 3, 3]
embed_dropout = 0.4
#####################################


#  Prepare the datasets
train_data = TensorDataset(
    torch.tensor(X_time_train),
    torch.tensor(X_tabular_train),
    torch.tensor(X_tabular_cat_train),
    torch.tensor(y_target_train[:, :output_weeks]),
)
valid_data = TensorDataset(
    torch.tensor(X_time_valid),
    torch.tensor(X_tabular_validation),
    torch.tensor(X_tabular_cat_valid),
    torch.tensor(y_target_valid[:, :output_weeks]),
)

# DataLoaders with sampler for training and default for validation
train_loader = DataLoader(
    train_data, batch_size=batch_size, drop_last=False
)

valid_loader = DataLoader(
    valid_data, shuffle=False, batch_size=batch_size, drop_last=False
)

test_data = TensorDataset(
    torch.tensor(X_time_test),
    torch.tensor(X_tabular_test),
    torch.tensor(X_tabular_cat_test),
    torch.tensor(y_target_test[:, :output_weeks]),
)

test_loader = DataLoader(
    test_data, shuffle=False, batch_size=batch_size, drop_last=False
)

list_cat = [len(np.unique(X_tabular_cat_train[:,i])) + 1 for i in range(X_tabular_cat_train.shape[1])]

def ablation_study(ablation_tabular=False,
                   ablation_TS=False,
                   ablation_attention=False,
                   etiquette="",):
    model_name = etiquette
    writer = SummaryWriter(f"{ROOT_TENSORBOARD}{model_name}/")

    print(f"|||||||||||| {model_name} ||||||||||||| \n\n")

    torch.manual_seed(42)
    np.random.seed(42)

    static_dim = X_tabular_train.shape[-1]
    print(static_dim)

    model = DroughtNet(
        output_weeks,
        X_time_train.shape[-1],
        hidden_dim,
        n_layers,
        ffnn_layers,
        dropout,
        static_dim,
        list_cat,
        embed_dim,
        embed_dropout,
        ablation_TS=ablation_TS,
        ablation_tabular=ablation_tabular,
        ablation_attention=ablation_attention,
    )
    model.apply(reset_weights)
    print(model)
    model.to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs
    )
    counter = 0
    valid_loss_min = np.inf
    for i in range(epochs):
        h = model.init_hidden(batch_size)

        for k, (inputs, static, cat ,labels) in tqdm(
            enumerate(train_loader),
            desc=f"epoch {i+1}/{epochs}",
            total=len(train_loader),
        ):
            model.train()
            counter += 1
            if len(inputs) < batch_size:
                h = model.init_hidden(len(inputs))
            h = tuple([e.data for e in h])
            inputs, labels, static, cat = (
                inputs.to(device),
                labels.to(device),
                static.to(device),
                cat.to(device),
            )
            model.zero_grad()
            output, h = model(inputs, h, static, cat)
            loss = loss_function(output, labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                if k == len(train_loader) - 1 or k == (len(train_loader) - 1) // 2:
                    val_h = model.init_hidden(batch_size)
                    val_losses = []
                    model.eval()
                    labels = []
                    preds = []
                    raw_labels = []
                    raw_preds = []
                    for inp, stat, cat, lab in valid_loader:
                        if len(inp) < batch_size:
                            val_h = model.init_hidden(len(inp))
                        val_h = tuple([each.data for each in val_h])
                        inp, lab, stat, cat = inp.to(device), lab.to(device), stat.to(device), cat.to(device)
                        out, val_h = model(inp, val_h, stat, cat)
                        val_loss = loss_function(out, lab.float())
                        val_losses.append(val_loss.item())
                        for labs in lab:
                            labels.append([int(l.round()) for l in labs])
                            raw_labels.append([float(l) for l in labs])
                        for pred in out:
                            preds.append([int(p.round()) for p in pred])
                            raw_preds.append([float(p) for p in pred])
                    # log data
                    labels = np.array(labels)
                    preds = np.clip(np.array(preds), 0, 5)
                    raw_preds = np.array(raw_preds)
                    raw_labels = np.array(raw_labels)
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
                        writer.add_scalars("Loss(MSE)", {'train': loss,
                                                        'validation': log_dict[f"{w}validation_loss"]},
                                                        counter)
                        writer.add_scalars("F1(MSE)", {'macro': log_dict[f"{w}macro_f1"],
                                                    'micro': log_dict[f"{w}micro_f1"]},
                                                    counter)
                        writer.add_scalar("MAE", log_dict[f"{w}mae"],
                                        counter)
                        writer.add_scalar("Learning-Rate", log_dict["lr"],
                                        counter)
                        for j, f1 in enumerate(
                            f1_score(labels[:, i], preds[:, i], average=None)
                        ):
                            log_dict[f"{w}{id2class[j]}_f1"] = f1
                        model.train()
                    if np.mean(val_losses) <= valid_loss_min:
                        torch.save(model.state_dict(), f"{ROOT_MODELS_WEIGHTS}{model_name}.pt")
                        print(
                            "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                                valid_loss_min, np.mean(val_losses)
                            )
                        )
                        valid_loss_min = np.mean(val_losses)
    
    with torch.no_grad():
        if k == len(train_loader) - 1 or k == (len(train_loader) - 1) // 2:
            test_h = model.init_hidden(batch_size)
            test_losses = []
            model.eval()
            labels = []
            preds = []
            raw_labels = []
            raw_preds = []
            for inp, stat, cat, lab in valid_loader:
                if len(inp) < batch_size:
                    test_h = model.init_hidden(len(inp))
                test_h = tuple([each.data for each in test_h])
                inp, lab, stat, cat = inp.to(device), lab.to(device), stat.to(device), cat.to(device)
                out, test_h = model(inp, test_h, stat, cat)
                pred = out.clone().detach()
                val_loss = loss_function(out, lab.float())
                test_losses.append(val_loss.item())

            dict_map = {"y_pred": [], "y_pred_rounded": [], "y_true": [], "week": []}
            for w in range(output_weeks):
                dict_map["y_pred"] += [float(p[w]) for p in pred]
                dict_map["y_pred_rounded"] += [int(p.round()[w]) for p in pred]
                dict_map["y_true"] += [float(item[w]) for item in lab]
                dict_map["week"] += [w] * len(inp)
            df = pd.DataFrame(dict_map)

            for w in range(6):
                wdf = df[df['week']==w]
                mae = mean_absolute_error(wdf['y_true'], wdf['y_pred']).round(3)
                f1 = f1_score(wdf['y_true'].round(),wdf['y_pred'].round(), average='macro').round(3)
                print(f"Week {w+1}", f"MAE {mae}", f"F1 {f1}")

            y_true_roc = df['y_true'].round()
            y_pred_roc = df['y_pred'].round()
            # y_pred_for_sklearn = np.array([[0, 0, 0, 0, 0, 0] for i in y_pred_roc])
            # for i in range(len(y_pred_roc)):
            #     y_pred_for_sklearn[i, int(y_pred_roc[i])] = 1

            # y_true_for_sklearn = np.array([[0, 0, 0, 0, 0, 0] for i in y_true_roc])
            # for i in range(len(y_true_roc)):
            #     y_true_for_sklearn[i, int(y_true_roc[i])] = 1


            mae = mean_absolute_error(df['y_true'], df['y_pred'])
            rmse = root_mean_squared_error(df['y_true'], df['y_pred'])
            f1 = f1_score(y_true_roc, y_pred_roc, average='macro')
            # roc_auc = roc_auc_score(y_true_for_sklearn, y_pred_for_sklearn, multi_class='ovr', average='weighted')

            # results = pd.DataFrame({'Model': ["Baseline"], 'MAE': [mae], 'RMSE': [rmse], 'F1': [f1], 'ROC_AUC': [roc_auc]})
            results = pd.DataFrame({'Model': [model_name], 'MAE': [mae], 'RMSE': [rmse], 'F1': [f1]})
            results.to_csv(f"{ROOT_RESULTS}/{model_name}.csv", index=False)

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

    # Define the ablation study
    entire = ablation_study(etiquette="entire")

    no_TS = ablation_study(ablation_TS=True,
                           etiquette="NO_TS"
                           )
    
    no_tab = ablation_study(ablation_tabular=True,
                            etiquette="NO_tabular",
                            )
    
    no_att = ablation_study(ablation_attention=True,
                            etiquette="NO_attention"
                            )
    
    no_tab_no_att = ablation_study(ablation_tabular=True,
                                ablation_attention=True,
                                etiquette="NO_tabular-NO_att",
                                )
    print(f"entire_val_loss: {entire}")
    print(f"no_TS_val_loss: {no_TS}")
    print(f"no_tab_no_att_val_loss: {no_tab_no_att}")
    print(f"no_tab_val_loss: {no_tab}")
    print(f"no_att_val_loss: {no_att}")