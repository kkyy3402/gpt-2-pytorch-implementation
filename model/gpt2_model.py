# model/gpt2_model.py

import torch
import torch.nn as nn
import math

# FlashAttention 설치 여부 확인
try:
    from flash_attn import flash_attn_func
    flash_attention_available = True
    print("### FlashAttention을 사용합니다.")
except ImportError:
    flash_attention_available = False
    print("### FlashAttention이 설치되어 있지 않습니다. 기본 Attention을 사용합니다.")

flash_attention_available = True

class GPT2Attention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        assert embed_size % num_heads == 0, "Embedding size must be divisible by num_heads"

        # Query, Key, Value 생성 (3 * embed_size)
        self.qkv_proj = nn.Linear(embed_size, 3 * embed_size)
        self.out_proj = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

        # FlashAttention이 설치되어 있으면 이를 사용하고, 그렇지 않으면 기본 Attention 사용
        if flash_attention_available:
            self.attention = flash_attn_func  # FlashAttention Function
        else:
            self.attention = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)

    def forward(self, x, mask=None):
        batch_size, seq_length, embed_size = x.size()

        # Query, Key, Value 생성
        qkv = self.qkv_proj(x)  # (batch_size, seq_length, 3 * embed_size)
        qkv = qkv.view(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, 3 * head_dim)

        q, k, v = qkv.chunk(3, dim=-1)  # Query, Key, Value 분리 (각각 (batch_size, num_heads, seq_length, head_dim))

        # print(f"qkv.shape : {qkv.shape}")

        if flash_attention_available:
            # FlashAttention 사용
            attn_output = self.attention(q, k, v, causal=True)  # FlashAttnFunc.forward를 적용
            attn_output = attn_output.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_length, embed_size)
        else:
            # 기존 PyTorch MultiheadAttention 사용: (seq_length, batch_size, embed_size) 형식으로 변환 필요
            q = q.permute(2, 0, 1, 3).contiguous().view(seq_length, batch_size, -1)  # (seq_length, batch_size, num_heads * head_dim)
            k = k.permute(2, 0, 1, 3).contiguous().view(seq_length, batch_size, -1)  # (seq_length, batch_size, num_heads * head_dim)
            v = v.permute(2, 0, 1, 3).contiguous().view(seq_length, batch_size, -1)  # (seq_length, batch_size, num_heads * head_dim)
            attn_output, _ = self.attention(q, k, v)
            attn_output = attn_output.permute(1, 0, 2).contiguous().view(batch_size, seq_length, embed_size)

        # 원래의 임베딩 크기로 변환
        # print(f"attn_output.shape : {attn_output.shape}")
        

        return self.out_proj(attn_output)

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
        self.attention = GPT2Attention(embed_size, num_heads)
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
