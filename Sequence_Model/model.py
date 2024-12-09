"""
Author: Sandeep Kumar Suresh
        EE23S059

Contains all the model definitions

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# import transformers
# from dataclasses import dataclass,field

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity 1
        self.relu1 = nn.ReLU()

        # Linear function 2: 100 --> 100
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 2
        self.relu2 = nn.ReLU()

        # Linear function 3: 100 --> 100
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 3
        self.relu3 = nn.ReLU()

        # Linear function 4 (readout): 100 --> 10
        self.fc4 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        # Linear function  # LINEAR
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)

        # Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.relu2(out)

        # Linear function 2
        out = self.fc3(out)
        # Non-linearity 2
        out = self.relu3(out)

        # Linear function 4 (readout)
        out = self.fc4(out)
        return out

    def initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            # Initialize weights with Xavier uniform
            torch.nn.init.xavier_uniform_(layer.weight)
            # Initialize biases with zeros
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

# @dataclass
# class TransformerConfig:
#     size: str
#     input_size: int = 134
#     max_position_embeddings: int = field(default=256, repr=False)
#     layer_norm_eps: float = field(default=1e-12, repr=False)
#     hidden_dropout_prob: float = field(default=0.1, repr=False)
#     hidden_size: int = field(default=512, repr=False)
#     num_attention_heads: int = field(default=8, repr=False)
#     num_hidden_layers: int = field(default=4, repr=False)
#     model_config: transformers.BertConfig = field(init=False)

#     def __post_init__(self):
#         assert self.size in ["small", "large"]
#         if self.size == "small":
#             self.hidden_size = 256
#             self.num_attention_heads = 4
#             self.num_hidden_layers = 2

#         self.model_config = transformers.BertConfig(
#             hidden_size=self.hidden_size,
#             num_attention_heads=self.num_attention_heads,
#             num_hidden_layers=self.num_hidden_layers,
#             max_position_embeddings=self.max_position_embeddings,
#         )


# class PositionEmbedding(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.position_embeddings = nn.Embedding(
#             config.max_position_embeddings, config.hidden_size
#         )
#         self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.register_buffer(
#             "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
#         )
#         self.position_embedding_type = getattr(
#             config, "position_embedding_type", "absolute"
#         )

#     def forward(self, x):
#         input_shape = x.size()
#         seq_length = input_shape[1]
#         position_ids = self.position_ids[:, :seq_length]

#         position_embeddings = self.position_embeddings(position_ids)
#         embeddings = x + position_embeddings
#         embeddings = self.LayerNorm(embeddings)
#         embeddings = self.dropout(embeddings)
#         return embeddings


# class Transformer(nn.Module):
#     def __init__(self, config, n_classes=50):
#         super().__init__()
#         self.l1 = nn.Linear(
#             in_features=config.input_size, out_features=config.hidden_size
#         )
#         self.embedding = PositionEmbedding(config)
#         self.layers = nn.ModuleList(
#             [
#                 transformers.BertLayer(config.model_config)
#                 for _ in range(config.num_hidden_layers)
#             ]
#         )
#         self.l2 = nn.Linear(in_features=config.hidden_size, out_features=n_classes)

#     def forward(self, x):
#         x = self.l1(x)
#         x = self.embedding(x)
#         for layer in self.layers:
#             x = layer(x)[0]

#         x = torch.max(x, dim=1).values
#         x = F.dropout(x, p=0.2)
#         x = self.l2(x)
#         return x


# class Custom_Model(nn.Module):
#     def __init__(self, pretrained_model, start_layer, end_layer):
#         super(Custom_Model, self).__init__()
#         self.start_layer = start_layer
#         self.pretrained_model = pretrained_model
#         self.end_layer = end_layer

#     def forward(self, x):
#         x = self.start_layer(x)        
#         x = self.pretrained_model(x)   
#         x = self.end_layer(x)            
#         return x



class FeedforwardNeuralNetModelLeftHand(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_prob = 0.5):
        """
        input_dim: Size of the input layer.
        hidden_dims: A list of integers where each integer specifies the number of neurons in that hidden layer.
        output_dim: Size of the output layer.
        dropout_prob: Dropout probability.
        """
        super(FeedforwardNeuralNetModelLeftHand, self).__init__()
        
        # Create lists to hold the layers and dropouts dynamically
        self.fc_layers = nn.ModuleList()
        self.relu_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        # First hidden layer (input_dim -> first hidden layer size)
        self.fc_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.relu_layers.append(nn.ReLU())
        self.dropout_layers.append(nn.Dropout(p=dropout_prob))

        # Create hidden layers dynamically (hidden_dims[i-1] -> hidden_dims[i])
        for i in range(1, len(hidden_dims)):
            self.fc_layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.relu_layers.append(nn.ReLU())
            self.dropout_layers.append(nn.Dropout(p=dropout_prob))

        # Output layer (last hidden layer -> output_dim)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        # Apply fixed weight initialization
        self.apply(self.initialize_weights)

    def forward(self, x):
        # Pass through each hidden layer with activation and dropout
        # print(x.shape)
        for fc, relu, dropout in zip(self.fc_layers, self.relu_layers, self.dropout_layers):
            x = fc(x)
            x = relu(x)
            x = dropout(x)

        # Output layer (no activation here, since it could be for classification or regression)
        x = self.output_layer(x)
        return x

    def initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            # Initialize weights with Xavier uniform
            torch.nn.init.xavier_uniform_(layer.weight)
            # Initialize biases with zeros
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)








# class Transformer(nn.Module):
#     def __init__(self, config, n_classes=10):
#         super().__init__()
#         self.l1 = nn.Linear(
#             in_features=config.input_size, out_features=config.hidden_size
#         )
#         self.embedding = PositionEmbedding(config)
#         self.layers = nn.ModuleList(
#             [
#                 transformers.BertLayer(config.model_config)
#                 for _ in range(config.num_hidden_layers)
#             ]
#         )
#         self.l2 = nn.Linear(in_features=config.hidden_size,out_features=config.hidden_size)
#         self.l3 = nn.Linear(in_features=config.hidden_size, out_features=n_classes)

#         self.freeze_layers()


#     def freeze_layers(self):
#         """
#         Modify the below code based on which layers to freeze
#         """
#         # Freeze l1, embedding, and layers (BERT layers) and making l2 and l3 trainable
#         for param in self.l1.parameters():
#             param.requires_grad = False

#         for param in self.embedding.parameters():
#             param.requires_grad = False

#         for layer in self.layers:
#             for param in layer.parameters():
#                 param.requires_grad = False
        
#         for param in self.l2.parameters():
#             param.requires_grad = True

#         for param in self.l3.parameters():
#             param.requires_grad = True





#     def forward(self, x):
#         x = self.l1(x)
#         x = self.embedding(x)
#         for layer in self.layers:
#             x = layer(x)[0]

#         x = torch.max(x, dim=1).values
#         x = F.dropout(x, p=0.2)
#         x = F.relu(self.l2(x))
#         x = self.l3(x)
#         return x
