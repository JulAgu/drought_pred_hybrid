import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HybridModel(nn.Module):
    """
    A hybrid architecture for processing heterogeneous data: tabular and time series.
    It uses an LSTM for time series.
    It uses embeddings for categorical variables.
    At the output of the LSTM, an attention layer is used to weight the outputs.
    Finally, the outputs of the tabular and temporal parts are concatenated and passed through linear layers.

    Attributes:
    ----------
        num_categorical_features: number of categorical features.
        list_unic_cat: list of unique values for each categorical feature.
        num_numerical_features: number of numerical features.
        num_time_series_features: number of features in the time series.
        hidden_size: size of LSTM's hidden states.
        num_lstm_layers: number of stacked LSTMs.
        dropout: dropout rate.
        embedding_dims: embedding size for categorical features.
        num_fc_tabular_layers: number of Fully Connected layers for tabular data.
        num_fc_combined_layers: number of Fully Connected layers for the combined representation.

    Methods:
    --------
        __forward__: method for passing data through the model.
    """

    def __init__(
        self,
        list_unic_cat,
        num_numerical_features,
        num_time_series_features,
        hidden_size,
        num_lstm_layers,
        dropout,
        embedding_dims,
        num_fc_tabular_layers,
        num_fc_combined_layers,
        output_size,
        ablation_TS=False,
        ablation_tabular=False,
        ablation_attention=False,
        ablation_embedding=False, # TODO: add ablation for embeddings, for this it is necessary to review the data loading process...
    ):
        super(HybridModel, self).__init__()
        
        self.ablation_tabular = ablation_tabular
        self.ablation_TS = ablation_TS
        self.ablation_attention = ablation_attention
        self.num_lstm_layers = num_lstm_layers
        self.hidden_size = hidden_size

        if not self.ablation_tabular:
            # Embeddings for categorical variables
            self.embeddings = nn.ModuleList(
                [
                    nn.Embedding(num_embeddings=i, embedding_dim=dimension)
                    for i, dimension in zip(list_unic_cat, embedding_dims)
                ]
            )

            total_embedding_dim = int(sum(embedding_dims))
            # Static data branch
            tabular_fc_layers = []
            input_size = total_embedding_dim + num_numerical_features
            for _ in range(num_fc_tabular_layers):
                tabular_fc_layers.append(nn.Linear(input_size, 512))
                tabular_fc_layers.append(nn.ReLU())
                input_size = 512
            self.tabular_fc_layers = nn.Sequential(
                *tabular_fc_layers, nn.Linear(512, 256), nn.ReLU()
            )

        if not self.ablation_TS:
            # TS branch
            self.lstm = nn.LSTM(
                input_size=num_time_series_features,
                hidden_size=hidden_size,
                num_layers=num_lstm_layers,
                dropout=dropout,
                batch_first=True,
            )

            # Atenttion
            self.attention = nn.Linear(hidden_size, 1)
            self.dropout = nn.Dropout(dropout)

        # Combined part
        self.fc_after_context = nn.Linear(hidden_size, 256)
        combined_fc_layers = []
        if not self.ablation_tabular and not self.ablation_TS:
            input_dim = 256 + 256  # Assuming 64 from tabular output and 64 from LSTM output after attention
        else:
            input_dim = 256
        for _ in range(num_fc_combined_layers):
            combined_fc_layers.append(nn.Linear(input_dim, 256))
            combined_fc_layers.append(nn.ReLU())
            input_dim = 256
        self.combined_fc_layers = nn.Sequential(
            *combined_fc_layers, nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, output_size)
        )

    def forward(self, numerical_data, categorical_data, time_series_data, hidden):
        if not self.ablation_tabular:
            # Embeddings for categorical data
            embeddings = [emb(categorical_data[:, i]) for i, emb in enumerate(self.embeddings)]
            x_cat = torch.cat(embeddings, dim=1)

            # Concatenate categorical and numerical data
            x_tabular = torch.cat((x_cat, numerical_data), dim=1)

            # Pass the tabular data through FC layers
            x1 = self.tabular_fc_layers(x_tabular)
        if not self.ablation_TS:
            # Pass the time series data through the LSTM
            lstm_out, hidden = self.lstm(time_series_data, hidden)
            # Pass the data through the attention mechanism
            if not self.ablation_attention:
                attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
                context_vector = torch.sum(attention_weights * lstm_out, dim=1)
                droped_out = self.dropout(context_vector)

                x2 = torch.relu(self.fc_after_context(droped_out))
            else:
                droped_out = self.dropout(lstm_out[:, -1, :])
                x2 = torch.relu(self.fc_after_context(droped_out))

        # Concatenate the outputs from the tabular and the temporal data and pass it through FC layers
        if not self.ablation_tabular and not self.ablation_TS:
            x = torch.cat((x1, x2), dim=1)
        elif not self.ablation_tabular:
            x = x1
        else:
            x = x2

        x = self.combined_fc_layers(x)
        return x, hidden


    def init_hidden(self, batch_size, device="cuda"):
        weight = next(self.lstm.parameters()).data
        hidden = (
            weight.new(self.num_lstm_layers, batch_size, self.hidden_size).zero_().to(device),
            weight.new(self.num_lstm_layers, batch_size, self.hidden_size).zero_().to(device),
        )
        return hidden
    

class HybridModel_2Outputs(nn.Module):
    """
    An alternative form of the HybridModel, where a multi task learning approach is uses to predict two outputs =
    The continual drought score and the categorical drought class.

    The idea is to combine CrossEntropyLoss and HuberLoss to train the model.
    I'm not completly sure of what form the output should have, but I think it should be a tuple of two tensors.
    As I would like to evaluate the 6 categories over the 6 weeks with cross entropy loss, I think the output should be 36 values.

    """

    def __init__(
        self,
        num_categorical_features,
        list_unic_cat,
        num_numerical_features,
        num_time_series_features,
        hidden_size,
        num_lstm_layers,
        dropout,
        embedding_dims,
        num_fc_tabular_layers,
        num_fc_combined_layers,
        output_size,
        ablation_TS=False,
        ablation_tabular=False,
        ablation_attention=False,
    ):
        super(HybridModel_2Outputs, self).__init__()
        self.output_size = output_size
        self.ablation_tabular = ablation_tabular
        self.ablation_TS = ablation_TS
        self.ablation_attention = ablation_attention

        if not self.ablation_tabular:
            # Embeddings for categorical variables
            self.embeddings = nn.ModuleList(
                [
                    nn.Embedding(num_embeddings=i, embedding_dim=embedding_dims)
                    for i in list_unic_cat
                ]
            )

            total_embedding_dim = num_categorical_features * embedding_dims

            # Static data branch
            tabular_fc_layers = []
            input_size = total_embedding_dim + num_numerical_features
            for _ in range(num_fc_tabular_layers):
                tabular_fc_layers.append(nn.Linear(input_size, 128))
                tabular_fc_layers.append(nn.ReLU())
                input_size = 128
            self.tabular_fc_layers = nn.Sequential(
                *tabular_fc_layers, nn.Linear(128, 64), nn.ReLU()
            )

        if not self.ablation_TS:
            # TS branch
            self.lstm = nn.LSTM(
                input_size=num_time_series_features,
                hidden_size=hidden_size,
                num_layers=num_lstm_layers,
                batch_first=True,
            )

            # Atenttion
            self.attention = nn.Linear(hidden_size, 1)
            self.dropout = nn.Dropout(dropout)

        # Combined part
        self.fc_after_context = nn.Linear(hidden_size, 64)
        combined_fc_layers = []
        if not self.ablation_tabular and not self.ablation_TS:
            input_dim = 64 + 64  # Assuming 64 from tabular output and 64 from LSTM output after attention
        else:
            input_dim = 64
        tmp_input_dim = input_dim
        for _ in range(num_fc_combined_layers):
            combined_fc_layers.append(nn.Linear(tmp_input_dim, 64))
            combined_fc_layers.append(nn.ReLU())
            tmp_input_dim = 64
        self.combined_fc_layers = nn.Sequential(
            *combined_fc_layers, nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, self.output_size)
        )

        self.categories_output = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, self.output_size*6)
        )


    def forward(self, categorical_data, numerical_data, time_series_data):
        numerical_data = numerical_data.to(torch.float32)
        time_series_data = time_series_data.to(torch.float32)
        if not self.ablation_tabular:
            # Embeddings for categorical data
            embeddings = [
                emb(categorical_data[:, i]) for i, emb in enumerate(self.embeddings)
            ]
            x_cat = torch.cat(embeddings, dim=1)

            # Concatenate categorical and numerical data
            x_tabular = torch.cat((x_cat, numerical_data), dim=1)

            # Pass the tabular data through FC layers
            x1 = self.tabular_fc_layers(x_tabular)
        if not self.ablation_TS:
            # Pass the time series data through the LSTM
            lstm_out, (hn, cn) = self.lstm(time_series_data)
            # Pass the data through the attention mechanism
            if not self.ablation_attention:
                attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
                context_vector = torch.sum(attention_weights * lstm_out, dim=1)
                droped_out = self.dropout(context_vector)

                x2 = torch.relu(self.fc_after_context(droped_out))
            else:
                droped_out = self.dropout(lstm_out[:, -1, :])
                x2 = torch.relu(self.fc_after_context(droped_out))

        # Concatenate the outputs from the tabular and the temporal data and pass it through FC layers
        if not self.ablation_tabular and not self.ablation_TS:
            x = torch.cat((x1, x2), dim=1)
        elif not self.ablation_tabular:
            x = x1
        else:
            x = x2

        output_class = self.categories_output(x)
        output_class = output_class.view(-1, 6, self.output_size,)
        output = self.combined_fc_layers(x)

        return output, output_class

class HybridModel_2Outputs_noEmbbedings(nn.Module):
    """
    Like HybridModel_2Outputs but without embeddings for categorical variables and including clip_norm for the gradients and dropout. 

    """

    def __init__(
        self,
        num_numerical_features,
        num_time_series_features,
        hidden_size,
        num_lstm_layers,
        dropout,
        num_fc_tabular_layers,
        num_fc_combined_layers,
        output_size,
        ablation_TS=False,
        ablation_tabular=False,
        ablation_attention=False,
    ):
        super(HybridModel_2Outputs_noEmbbedings, self).__init__()
        self.output_size = output_size
        self.ablation_tabular = ablation_tabular
        self.ablation_TS = ablation_TS
        self.ablation_attention = ablation_attention

        if not self.ablation_tabular:

            # Static data branch
            tabular_fc_layers = []
            input_size = num_numerical_features
            for _ in range(num_fc_tabular_layers):
                tabular_fc_layers.append(nn.Linear(input_size, 128))
                tabular_fc_layers.append(nn.ReLU())
                input_size = 128
            self.tabular_fc_layers = nn.Sequential(
                *tabular_fc_layers, nn.Linear(128, 64), nn.ReLU()
            )

        if not self.ablation_TS:
            # TS branch
            self.lstm = nn.LSTM(
                input_size=num_time_series_features,
                hidden_size=hidden_size,
                num_layers=num_lstm_layers,
                dropout=dropout,
                batch_first=True,
            )

            # Atenttion
            self.attention = nn.Linear(hidden_size, 1)
            self.dropout = nn.Dropout(dropout)

        # Combined part
        self.fc_after_context = nn.Linear(hidden_size, 64)
        combined_fc_layers = []
        if not self.ablation_tabular and not self.ablation_TS:
            input_dim = 64 + 64  # Assuming 64 from tabular output and 64 from LSTM output after attention
        else:
            input_dim = 64
        tmp_input_dim = input_dim
        for _ in range(num_fc_combined_layers):
            combined_fc_layers.append(nn.Linear(tmp_input_dim, 64))
            combined_fc_layers.append(nn.ReLU())
            tmp_input_dim = 64
        self.combined_fc_layers = nn.Sequential(
            *combined_fc_layers, nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, self.output_size)
        )

        self.categories_output = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, self.output_size*6)
        )


    def forward(self, time_series_data, numerical_data):
        numerical_data = numerical_data.to(torch.float32)
        time_series_data = time_series_data.to(torch.float32)
        if not self.ablation_tabular:
            x1 = self.tabular_fc_layers(numerical_data)
        if not self.ablation_TS:
            # Pass the time series data through the LSTM
            lstm_out, (hn, cn) = self.lstm(time_series_data)
            # Pass the data through the attention mechanism
            if not self.ablation_attention:
                attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
                context_vector = torch.sum(attention_weights * lstm_out, dim=1)
            else:
                context_vector = lstm_out[:, -1, :]  # Last time step output
            
            droped_out = self.dropout(context_vector)
            x2 = torch.relu(self.fc_after_context(droped_out))

        # Concatenate the outputs from the tabular and the temporal data and pass it through FC layers
        if not self.ablation_tabular and not self.ablation_TS:
            x = torch.cat((x1, x2), dim=1)
        elif not self.ablation_tabular:
            x = x1
        else:
            x = x2

        output_class = self.categories_output(x)
        output_class = output_class.view(-1, 6, self.output_size,)
        output = self.combined_fc_layers(x)

        return output, output_class


class HybridModel_custom(nn.Module):
    def __init__(
        self,
        num_numerical_features,
        num_time_series_features,
        hidden_size,
        num_lstm_layers,
        num_fc_tabular_layers,
        num_fc_combined_layers,
        output_size,
        dropout,
        ablation_TS=False,
        ablation_tabular=False,
        ablation_attention=False,
    ):
        super(HybridModel_custom, self).__init__()
        
        self.ablation_tabular = ablation_tabular
        self.ablation_TS = ablation_TS
        self.ablation_attention = ablation_attention

        if not self.ablation_tabular:
            # Static data branch
            tabular_fc_layers = []
            input_size = num_numerical_features
            for _ in range(num_fc_tabular_layers):
                tabular_fc_layers.append(nn.Linear(input_size, 128))
                tabular_fc_layers.append(nn.ReLU())
                input_size = 128
            self.tabular_fc_layers = nn.Sequential(
                *tabular_fc_layers, nn.Linear(128, 64), nn.ReLU()
            )

        if not self.ablation_TS:
            # TS branch
            self.lstm = nn.LSTM(
                input_size=num_time_series_features,
                hidden_size=hidden_size,
                num_layers=num_lstm_layers,
                dropout=dropout,
                batch_first=True,
            )

            # Atenttion
            self.attention = nn.Linear(hidden_size, 1)
            self.dropout = nn.Dropout(dropout)

        # Combined part
        self.fc_after_context = nn.Linear(hidden_size, 64)
        combined_fc_layers = []
        if not self.ablation_tabular and not self.ablation_TS:
            input_dim = 64 + 64  # Assuming 64 from tabular output and 64 from LSTM output after attention
        else:
            input_dim = 64
        for _ in range(num_fc_combined_layers):
            combined_fc_layers.append(nn.Linear(input_dim, 64))
            combined_fc_layers.append(nn.ReLU())
            input_dim = 64
        self.combined_fc_layers = nn.Sequential(
            *combined_fc_layers, nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, output_size)
        )

    def forward(self, time_series_data, numerical_data):
        numerical_data = numerical_data.to(torch.float32)
        time_series_data = time_series_data.to(torch.float32)
        if not self.ablation_tabular:
            # Pass the tabular data through FC layers
            x1 = self.tabular_fc_layers(numerical_data)
        if not self.ablation_TS:
            # Pass the time series data through the LSTM
            lstm_out, (hn, cn) = self.lstm(time_series_data)
            # Pass the data through the attention mechanism
            if not self.ablation_attention:
                attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
                context_vector = torch.sum(attention_weights * lstm_out, dim=1)
            else:
                context_vector = lstm_out[:, -1, :]  # Last time step output
            
            droped_out = self.dropout(context_vector)
            x2 = torch.relu(self.fc_after_context(droped_out))

        # Concatenate the outputs from the tabular and the temporal data and pass it through FC layers
        if not self.ablation_tabular and not self.ablation_TS:
            x = torch.cat((x1, x2), dim=1)
        elif not self.ablation_tabular:
            x = x1
        else:
            x = x2
        x = self.combined_fc_layers(x)
        return x

class New_HybridModel(nn.Module):
    """
    The last version of the HybridModel, with the same architecture but smalle changes in the forward method and in the embeddings traitement.
    """
    def __init__(
        self,
        num_numerical_features,
        num_time_series_features,
        hidden_size,
        num_lstm_layers,
        list_unic_cat,
        embedding_dims,
        num_fc_tabular_layers,
        num_fc_combined_layers,
        output_size,
        dropout,
        ablation_TS=False,
        ablation_tabular=False,
        ablation_attention=False,
    ):
        super(New_HybridModel, self).__init__()
        self.num_lstm_layers = num_lstm_layers
        self.hidden_size = hidden_size

        self.embeddings = nn.ModuleList(
                [
                    nn.Embedding(num_embeddings=i, embedding_dim=dimension)
                    for i, dimension in zip(list_unic_cat, embedding_dims)
                ]
            )

        self.after_embeddings = nn.Sequential(
            nn.Linear(int(sum(embedding_dims)), 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 16)
        )

        tabular_total_size = num_numerical_features + 16
        tabular_fc_layers = []
        for _ in range(num_fc_tabular_layers):
            tabular_fc_layers.append(nn.Linear(tabular_total_size, tabular_total_size))
            tabular_fc_layers.append(nn.ReLU())
        self.tabular_fc_layers = nn.Sequential(
            *tabular_fc_layers, nn.Linear(tabular_total_size, tabular_total_size)
        )

        # TS branch
        self.lstm = nn.LSTM(
            input_size=num_time_series_features,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.attention = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

        combined_fc_layers = []
        input_dim = tabular_total_size + hidden_size

        for _ in range(num_fc_combined_layers):
            combined_fc_layers.append(nn.Linear(input_dim, hidden_size))
            combined_fc_layers.append(nn.ReLU())
            input_dim = hidden_size
        self.combined_fc_layers = nn.Sequential(
            *combined_fc_layers, nn.Linear(hidden_size, output_size)
        )

    def forward(self, time_series_data, numerical_data, categorical_data):
        batch_size = time_series_data.size(0)
        h0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size).to(device)

        time_series_data = time_series_data.to(torch.float32)
        numerical_data = numerical_data.to(torch.float32)
        categorical_data = categorical_data.to(torch.int64)

        embeddings = [emb(categorical_data[:, i]) for i, emb in enumerate(self.embeddings)]
        x_cat = torch.cat(embeddings, dim=1)
        x_cat = self.after_embeddings(x_cat)
        x_tabular = torch.cat((x_cat, numerical_data), dim=1)
        x1 = self.tabular_fc_layers(x_tabular)

        # Pass the time series data through the LSTM and the attention mechanism
        lstm_out, _ = self.lstm(time_series_data, (h0, c0))
        lstm_out = self.layer_norm(lstm_out) # Apply layer normalization
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        # Pass the data through the attention mechanism
        # context_vector = lstm_out[:, -1, :]  # Last time step output

        # Combined MLPs and output
        x2 = self.dropout(context_vector)
        x = torch.cat((x1, x2), dim=1)
        x = self.combined_fc_layers(x)
        return x