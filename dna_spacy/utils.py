import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


from torch.utils.data import Dataset, DataLoader
import datetime
import os


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
    


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        # nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(self.device)

        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output


    def encode(self, test_data):
        test_data = test_data.to(self.device)

        with torch.no_grad():
            self.eval()  # Set the model to evaluation mode

            # Get the encoding layer output
            src_mask, _ = self.generate_mask(test_data, test_data[:, :-1])
            src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(test_data)))

            encoding_layer_output = src_embedded
            for enc_layer in self.encoder_layers:
                encoding_layer_output = enc_layer(encoding_layer_output, src_mask)

            # Average the encoding vectors for each data point
            averaged_vectors = torch.mean(encoding_layer_output, dim=1)

        # Flatten the averaged vectors and move to cpu
        flattened_vectors = averaged_vectors.view(-1, 256).cpu().numpy()

        return flattened_vectors

    
class MyDataset(Dataset):
    def __init__(self, df):
        self.data = np.array(df.window.tolist())
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src = self.data[idx]
        tgt = self.data[idx]
        return src, tgt
    

import os
import torch
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader
import datetime

class TransformerTrainer:
    def __init__(self,
                 train_dataset,
                 val_dataset,

                 src_vocab_size = 16400, 
                 tgt_vocab_size = 16400, 
                 d_model = 256, 
                 num_heads = 8, 
                 num_layers = 6, 
                 d_ff = 2048, 
                 max_seq_length = 14, 
                 dropout = 0.1, 
                 batch_size = 16, 
                 lr = 0.00001, 
                 num_epochs = 10, 
                 save = False):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.save = save

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
        self.transformer.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = optim.Adam(self.transformer.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9)
        self.train_losses = []
        self.val_losses = []
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def load_data(self):
        self.dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)

    def train_epoch(self, epoch):
        epoch_train_loss = 0.0
        for src_data, target_data in self.dataloader:
            src_data = src_data.to(self.device)
            target_data = target_data.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.transformer(src_data, target_data[:, :-1])
            loss = self.criterion(output.contiguous().view(-1, self.tgt_vocab_size), target_data[:, 1:].contiguous().view(-1))
            loss.backward()
            self.optimizer.step()
            
            epoch_train_loss += loss.item()
        
        self.train_losses.append(epoch_train_loss / len(self.dataloader))

    def validate_epoch(self, epoch):
        epoch_val_loss = 0.0
        self.transformer.eval()  
        with torch.no_grad():  
            for src_data, target_data in self.val_dataloader:
                src_data = src_data.to(self.device)
                target_data = target_data.to(self.device)

                output = self.transformer(src_data, target_data[:, :-1])
                loss = self.criterion(output.contiguous().view(-1, self.tgt_vocab_size), target_data[:, 1:].contiguous().view(-1))

                epoch_val_loss += loss.item()

        self.val_losses.append(epoch_val_loss / len(self.val_dataloader))
        self.transformer.train()

    def save_model(self, epoch):
        if self.save:
            os.makedirs(self.timestamp, exist_ok=True)
            epoch_str = str(epoch+1).zfill(2) 
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.transformer.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.train_losses[-1],
                'src_vocab_size': self.src_vocab_size,
                'tgt_vocab_size': self.tgt_vocab_size,
                'd_model': self.d_model,
                'num_heads': self.num_heads,
                'num_layers': self.num_layers,
                'd_ff': self.d_ff,
                'max_seq_length': self.max_seq_length,
                'dropout': self.dropout,
            }, os.path.join(self.timestamp, f"epoch_{epoch_str}.pth"))

    def train(self):
        self.load_data()
        for epoch in range(self.num_epochs):
            self.train_epoch(epoch)
            self.validate_epoch(epoch)
            print(f"Epoch: {epoch+1}, Training Loss: {self.train_losses[-1]:.6f}, Validation Loss: {self.val_losses[-1]:.6f}")
            self.save_model(epoch)

    def encode(self, test_data):
        self.transformer.eval()  # Set the model to evaluation mode

        with torch.no_grad():
            src_mask, _ = self.transformer.generate_mask(test_data, test_data[:, :-1])
            src_embedded = self.transformer.dropout(self.transformer.positional_encoding(self.transformer.encoder_embedding(test_data)))

            encoding_layer_output = src_embedded
            for enc_layer in self.transformer.encoder_layers:
                encoding_layer_output = enc_layer(encoding_layer_output, src_mask)

            averaged_vectors = torch.mean(encoding_layer_output, dim=1)

        flattened_vectors = averaged_vectors.view(-1, self.d_model).cpu().numpy()

        return flattened_vectors

class MyDataset(Dataset):
    def __init__(self, df):
        self.data = np.array(df.window.tolist())
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src = self.data[idx]
        tgt = self.data[idx]
        return src, tgt

def prepare_data(path, start, end):
    df = pd.read_feather(path)[start:end]
    df = df.sort_values('sequence_index')
    dataset = MyDataset(df)
    return dataset



def load_model(path_to_saved_model):
    checkpoint = torch.load(path_to_saved_model)

    # Initialize the model
    model = utils.Transformer(
        src_vocab_size=checkpoint['src_vocab_size'],
        tgt_vocab_size=checkpoint['tgt_vocab_size'],
        d_model=checkpoint['d_model'],
        num_heads=checkpoint['num_heads'],
        num_layers=checkpoint['num_layers'],
        d_ff=checkpoint['d_ff'],
        max_seq_length=checkpoint['max_seq_length'],
        dropout=checkpoint['dropout']
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode

    return model, device