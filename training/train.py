import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tokenizer.tokenizer import GPTTokenizer
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
        # 각 시퀀스는 self.seq_length만큼 슬라이딩 방식으로 가져옴
        input_tokens = self.tokens[idx:idx+self.seq_length]  # 입력 시퀀스
        target_tokens = self.tokens[idx+1:idx+self.seq_length+1]  # 다음 시퀀스 (shifted)

        # 길이 부족 시 예외 처리 (예: 마지막 시퀀스 부분)
        if len(input_tokens) < self.seq_length or len(target_tokens) < self.seq_length:
            raise IndexError("데이터의 끝에 도달했습니다.")
        
        # 텐서로 변환
        return (
            torch.tensor(input_tokens, dtype=torch.long),
            torch.tensor(target_tokens, dtype=torch.long)
        )


def load_data_from_file(tokenizer: GPTTokenizer, data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        tokens = f.read()
    return tokens

def load_data(tokenizer, config):
    tokens = load_data_from_file(tokenizer, config['data']['processed_path'])
    tokens = tokens.split()  # 띄어쓰기 기준으로 리스트화
    tokens = [int(token) for token in tokens]
    tokens = tokens[:3000]

    return split_dataset(tokens, config['training']['seq_length'], config['training']['batch_size'])

def split_dataset(tokens, seq_length, batch_size):
    dataset = TextDataset(tokens, seq_length)
    val_size = int(0.15 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=10)

    return train_loader, val_loader

def train_epoch(model, dataloader, criterion, optimizer, scaler, device, vocab_size):
    model.train()
    epoch_loss = 0
    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda'):
            outputs = model(inputs)  # (N, seq_length, vocab_size)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device, vocab_size):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validating"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)  # (N, seq_length, vocab_size)
                loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

            val_loss += loss.item()
    return val_loss / len(dataloader)

def save_model_if_best(model, avg_val_loss, best_val_loss, save_path):
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), save_path)
        print(f"모델이 검증 손실 {avg_val_loss:.4f}로 갱신되어 저장되었습니다.")
    return best_val_loss

def train_model(config):
    device = get_device(config)
    print(f"사용 디바이스: {device}")

    # 데이터 로드 및 분리
    tokenizer = GPTTokenizer(config)
    vocab_size = tokenizer.get_vocab_size()
    train_loader, val_loader = load_data(tokenizer, config)

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
    scaler = torch.amp.GradScaler()
    best_val_loss = float('inf')

    # 학습 루프
    for epoch in range(config['training']['epochs']):
        print(f"에포크 {epoch+1}/{config['training']['epochs']} 시작")
        
        # 학습
        avg_train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, vocab_size)
        print(f"에포크 {epoch+1} 완료. 평균 학습 손실: {avg_train_loss:.4f}")

        # 검증
        avg_val_loss = validate_epoch(model, val_loader, criterion, device, vocab_size)
        print(f"에포크 {epoch+1} 완료. 평균 검증 손실: {avg_val_loss:.4f}")

        # 모델 저장
        best_val_loss = save_model_if_best(model, avg_val_loss, best_val_loss, config['model']['save_path'])

if __name__ == "__main__":
    config = load_config()
    train_model(config)
