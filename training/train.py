# training/train.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tokenizer.tokenizer import SimpleTokenizer
from model.gpt2_model import GPT2
from utils.device_setup import get_device
from utils.config import load_config

class TextDataset(Dataset):
    def __init__(self, tokens, seq_length):
        self.tokens = tokens
        self.seq_length = seq_length

    def __len__(self):
        return len(self.tokens) - self.seq_length

    def __getitem__(self, idx):
        return (
            torch.tensor(self.tokens[idx:idx+self.seq_length], dtype=torch.long),
            torch.tensor(self.tokens[idx+1:idx+self.seq_length+1], dtype=torch.long)
        )

def load_data(tokenizer: SimpleTokenizer, data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    return tokens

def train_model(config):
    device = get_device(config)
    print(f"사용 디바이스: {device}")

    # 토크나이저 로드
    tokenizer = SimpleTokenizer()
    tokenizer.load(config['tokenizer']['save_path'])
    vocab_size = len(tokenizer.token_to_id)

    # 데이터 로드
    tokens = load_data(tokenizer, config['data']['processed_path'])
    dataset = TextDataset(tokens, config['training']['seq_length'])
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # 모델 초기화
    model = GPT2(
        vocab_size=vocab_size,
        embed_size=config['model']['embed_size'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        ff_hidden_dim=config['model']['ff_hidden_dim'],
        max_length=config['model']['max_length'],
        dropout=config['model']['dropout']
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']))

    # Mixed Precision Scaler 초기화
    scaler = torch.cuda.amp.GradScaler()

    # 학습 루프
    model.train()
    for epoch in range(config['training']['epochs']):
        epoch_loss = 0
        for inputs, targets in tqdm(dataloader, desc=f"에포크 {epoch+1}/{config['training']['epochs']}"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(inputs)  # (N, seq_length, vocab_size)
                loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"에포크 {epoch+1} 완료. 평균 손실: {avg_loss:.4f}")

    # 모델 저장
    torch.save(model.state_dict(), config['model']['save_path'])
    print(f"모델이 {config['model']['save_path']}에 저장되었습니다.")
