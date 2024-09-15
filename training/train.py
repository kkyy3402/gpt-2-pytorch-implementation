import os
import torch
import wandb  # wandb 라이브러리 추가
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
        input_tokens = self.tokens[idx:idx+self.seq_length]
        target_tokens = self.tokens[idx+1:idx+self.seq_length+1]
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
    # tokens = tokens[:3000]

    return split_dataset(tokens, config['training']['seq_length'], config['training']['batch_size'])

def split_dataset(tokens, seq_length, batch_size):
    dataset = TextDataset(tokens, seq_length)
    val_size = int(0.15 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=10)

    return train_loader, val_loader

def train_epoch(model, dataloader, criterion, optimizer, scaler, device, vocab_size, epoch):
    model.train()
    epoch_loss = 0
    for step, (inputs, targets) in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch+1}")):
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

        # 50 스텝마다 wandb로 학습 손실 기록
        if step % 50 == 0:
            wandb.log({"Train Loss": loss.item(), "Step": step, "Epoch": epoch+1})

    return epoch_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device, vocab_size, epoch):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(tqdm(dataloader, desc=f"Validating Epoch {epoch+1}")):
            inputs = inputs.to(device)
            targets = targets.to(device)

            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)  # (N, seq_length, vocab_size)
                loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

            val_loss += loss.item()

            # wandb로 검증 손실 기록
            wandb.log({"Validation Loss": loss.item(), "Step": step, "Epoch": epoch+1})

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

    # wandb 설정
    wandb.init(project=config["project_name"], config=config)

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

    # Early Stopping 설정
    early_stopping_patience = config['training'].get('early_stopping_patience', 5)  # 설정에서 patience를 받아오거나 기본값으로 5 사용
    early_stopping_counter = 0

    # 학습 루프
    for epoch in range(config['training']['epochs']):
        print(f"에포크 {epoch+1}/{config['training']['epochs']} 시작")
        
        # 학습
        avg_train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, vocab_size, epoch)
        print(f"에포크 {epoch+1} 완료. 평균 학습 손실: {avg_train_loss:.4f}")

        # 검증
        avg_val_loss = validate_epoch(model, val_loader, criterion, device, vocab_size, epoch)
        print(f"에포크 {epoch+1} 완료. 평균 검증 손실: {avg_val_loss:.4f}")

        # 모델 저장
        best_val_loss = save_model_if_best(model, avg_val_loss, best_val_loss, config['model']['save_path'])

        # Early Stopping 체크
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0  # 모델 성능이 개선되면 counter 초기화
        else:
            early_stopping_counter += 1
            print(f"Early stopping 카운터 증가: {early_stopping_counter}/{early_stopping_patience}")
        
        if early_stopping_counter >= early_stopping_patience:
            print(f"검증 손실이 {early_stopping_patience} 에포크 동안 개선되지 않아 학습을 중단합니다.")
            break

if __name__ == "__main__":
    config = load_config()
    train_model(config)
