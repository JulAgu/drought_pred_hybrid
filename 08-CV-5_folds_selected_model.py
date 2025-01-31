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
import utilities


class DroughtNetLSTM(nn.Module):
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
    ):
        super(DroughtNetLSTM, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim


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
        for i in range(ffnn_layers - 1):
            if i == 0:
                self.fflayers.append(nn.Linear(hidden_dim*2 + static_dim + 7, hidden_dim))
            else:
                self.fflayers.append(nn.Linear(hidden_dim, hidden_dim))
        self.fflayers = nn.ModuleList(self.fflayers)
        self.final = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden, static, cat):
        batch_size = x.size(0)
        x = x.to(dtype=torch.float32)
        static = static.to(dtype=torch.float32)
        lstm_out, hidden = self.lstm(x, hidden)
        last_hidden = lstm_out[:, -1, :]
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        att_out = torch.cat((context_vector, last_hidden), 1)
        out = self.dropout(att_out)

        embeddings = [emb(cat[:, i]) for i, emb in enumerate(self.embeddings)]
        cat = torch.cat(embeddings, dim=1)
        cat = self.embeddings_dropout(cat)
        cat = self.after_embeddings(cat)
        
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

def predict(x, val_h, static, cat):
    out, _ = model(torch.tensor(x), val_h, static, cat)
    return out


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

##### Hyperparameters #####
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
###########################


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

# Concat the datasets
dataset = ConcatDataset([train_data, valid_data, test_data])

K_folds = 5
kf = KFold(n_splits=K_folds, shuffle=True, random_state=42)

print("Starting the 5-fold CV")

for fold, (train_index, test_index) in enumerate(kf.split(dataset)):
    print(f"Fold {fold+1}")
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_index)

    # in_contex_dataloaders
    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_subsampler
    )

    test_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=test_subsampler
    )
    X_tabular_train = []
    X_tabular_categories_train = []
    for _, X_tab_train, X_tabular_cat_train, _ in train_loader:
        X_tabular_train.append(X_tab_train)
        X_tabular_categories_train.append(X_tabular_cat_train)
    X_tabular_train = torch.cat(X_tabular_train, dim=0).numpy()
    X_tabular_categories_train = torch.cat(X_tabular_categories_train, dim=0).numpy()

        

    list_cat = [len(np.unique(X_tabular_categories_train[:,i])) + 1 for i in range(X_tabular_categories_train.shape[1])]
    print(list_cat)

    # Now a classic training loop
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
        print("using GPU")
    else:
        device = torch.device("cpu")
        print("using CPU")

    static_dim = X_tabular_train.shape[-1]
    model = DroughtNetLSTM(
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
    )
    model.apply(reset_weights)
    model.to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs
    )
    counter = 0
    valid_loss_min = np.inf
    torch.manual_seed(42)
    np.random.seed(42)
    current_loss = 0.0
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

            # Print statistics
            current_loss += loss.item()
            if k % 100 == 99:
                print('EPOCH: %5d || Loss after mini-batch %5d: %.3f' %
                    (i + 1, k + 1, current_loss / 100))
                current_loss = 0.0

        torch.save(model.state_dict(), f"models/selected_{fold+1}.pt")

    with torch.no_grad():
        if k == len(train_loader) - 1 or k == (len(train_loader) - 1) // 2:
            val_h = model.init_hidden(batch_size)
            val_losses = []
            model.eval()
            labels = []
            preds = []
            raw_labels = []
            raw_preds = []
            for inp, stat, cat, lab in test_loader:
                if len(inp) < batch_size:
                    val_h = model.init_hidden(len(inp))
                val_h = tuple([each.data for each in val_h])
                inp, lab, stat, cat = inp.to(device), lab.to(device), stat.to(device), cat.to(device)
                out, val_h = model(inp, val_h, stat, cat)
                pred = out.clone().detach()
                val_loss = loss_function(out, lab.float())
                val_losses.append(val_loss.item())

            dict_map = {"y_pred": [], "y_pred_rounded": [], "y_true": [], "week": []}
            for w in range(output_weeks):
                dict_map["y_pred"] += [float(p[w]) for p in pred]
                dict_map["y_pred_rounded"] += [int(p.round()[w]) for p in pred]
                dict_map["y_true"] += [float(item[w]) for item in lab]
                dict_map["week"] += [w] * len(inp)
            df = pd.DataFrame(dict_map)
     
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
            validation_loss = np.mean(val_losses)  

            # results = pd.DataFrame({'Model': ["Baseline"], 'MAE': [mae], 'RMSE': [rmse], 'F1': [f1], 'ROC_AUC': [roc_auc], 'Validation Loss': [validation_loss]})
            results = pd.DataFrame({'Model': ["Baseline"], 'MAE': [mae], 'RMSE': [rmse], 'F1': [f1], 'Validation Loss': [validation_loss]})
            results.to_csv(f"results/selected_model_{fold+1}.csv", index=False)

                