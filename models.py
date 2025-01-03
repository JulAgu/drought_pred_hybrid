import torch
import torch.nn as nn

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
    ):
        super(HybridModel, self).__init__()

        # Embeddings for categorical variables
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=i, embedding_dim=embedding_dims)
                for i in list_unic_cat
            ]
        )

        total_embedding_dim = num_categorical_features * embedding_dims

        # Tabular part: dinamic creation of layers
        tabular_fc_layers = []
        input_size = total_embedding_dim + num_numerical_features
        for _ in range(num_fc_tabular_layers):
            tabular_fc_layers.append(nn.Linear(input_size, 128))
            tabular_fc_layers.append(nn.ReLU())
            input_size = 128
        self.tabular_fc_layers = nn.Sequential(
            *tabular_fc_layers, nn.Linear(128, 64), nn.ReLU()
        )

        # Temporal series
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
        input_dim = (64 + 64)  # Assuming 128 from tabular output and 64 from LSTM output after attention
        for _ in range(num_fc_combined_layers):
            combined_fc_layers.append(nn.Linear(input_dim, 64))
            combined_fc_layers.append(nn.ReLU())
            input_dim = 64
        self.combined_fc_layers = nn.Sequential(
            *combined_fc_layers, nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, output_size)
        )

    def forward(self, categorical_data, numerical_data, time_series_data):
        # Embeddings for categorical data
        embeddings = [
            emb(categorical_data[:, i]) for i, emb in enumerate(self.embeddings)
        ]
        x_cat = torch.cat(embeddings, dim=1)

        # Concatenate categorical and numerical data
        x_tabular = torch.cat((x_cat, numerical_data), dim=1)

        # Pass the tabular data through FC layers
        x1 = self.tabular_fc_layers(x_tabular)

        # Pass the time series data through the LSTM
        lstm_out, (hn, cn) = self.lstm(time_series_data)
        # Pass the data through the attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        droped_out = self.dropout(context_vector)

        x2 = torch.relu(self.fc_after_context(droped_out))

        # Concatenate the outputs from the tabular and the temporal data and pass it through FC layers
        x = torch.cat((x1, x2), dim=1)
        x = self.combined_fc_layers(x)

        return x
