# model/gpt2_model.py

import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(SelfAttention, self).__init__()
        assert embed_size % num_heads == 0, "임베딩 크기는 헤드 수로 나눠 떨어져야 합니다."
        
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        
        self.fc_out = nn.Linear(embed_size, embed_size)
    
    def forward(self, x, mask=None):
        N, seq_length, embed_size = x.shape
        # 쿼리, 키, 값 생성
        Q = self.query(x)  # (N, seq_length, embed_size)
        K = self.key(x)
        V = self.value(x)
        
        # 헤드 분할
        Q = Q.view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # (N, num_heads, seq_length, head_dim)
        K = K.view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 스케일된 닷 프로덕트 어텐션
        energy = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (N, num_heads, seq_length, seq_length)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy, dim=-1)  # (N, num_heads, seq_length, seq_length)
        
        out = torch.matmul(attention, V)  # (N, num_heads, seq_length, head_dim)
        
        out = out.transpose(1, 2).contiguous().view(N, seq_length, embed_size)  # (N, seq_length, embed_size)
        
        out = self.fc_out(out)  # (N, seq_length, embed_size)
        
        return out

class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_dim)
        self.fc2 = nn.Linear(ff_hidden_dim, embed_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_dim, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.ff = FeedForward(embed_size, ff_hidden_dim)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        att_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(att_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class GPT2(nn.Module):
    def __init__(self, vocab_size, embed_size=768, num_layers=12, num_heads=12, ff_hidden_dim=3072, max_length=512, dropout=0.1):
        super(GPT2, self).__init__()
        self.embed_size = embed_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size)
    
    def forward(self, x, mask=None):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).unsqueeze(0).expand(N, seq_length).to(x.device)
        x = self.token_embedding(x) + self.position_embedding(positions)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.layer_norm(x)
        logits = self.fc_out(x)
        return logits

def get_model(config, device):
    model = GPT2(
        vocab_size=config['tokenizer']['vocab_size'],
        embed_size=config['model']['embed_size'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        ff_hidden_dim=config['model']['ff_hidden_dim'],
        max_length=config['model']['max_length'],
        dropout=config['model']['dropout']
    ).to(device)
    return model

if __name__ == "__main__":
    # 예제 모델 생성
    from utils.config import load_config
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2(
        vocab_size=config['tokenizer']['vocab_size'],
        embed_size=config['model']['embed_size'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        ff_hidden_dim=config['model']['ff_hidden_dim'],
        max_length=config['model']['max_length'],
        dropout=config['model']['dropout']
    ).to(device)
    print(model)
