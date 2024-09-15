# interactive/interactive.py

import torch
from tokenizer.tokenizer import SimpleTokenizer
from model.gpt2_model import GPT2
from utils.device_setup import get_device
from utils.config import load_config

def load_model(config, device):
    vocab_size = config['tokenizer']['vocab_size']
    model = GPT2(
        vocab_size=vocab_size,
        embed_size=config['model']['embed_size'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        ff_hidden_dim=config['model']['ff_hidden_dim'],
        max_length=config['model']['max_length'],
        dropout=config['model']['dropout']
    ).to(device)
    model.load_state_dict(torch.load(config['model']['save_path'], map_location=device))
    model.eval()
    return model

def generate_text(model, tokenizer, device, prompt, max_length=100, temperature=1.0, top_k=50):
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    generated = input_ids
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(generated)
            logits = outputs[:, -1, :] / temperature
            # Top-K 샘플링
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            probabilities = torch.softmax(top_k_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            next_token = top_k_indices.gather(-1, next_token)
            generated = torch.cat((generated, next_token), dim=1)
    
    generated_text = tokenizer.decode(generated.squeeze().tolist())
    return generated_text

def interactive_mode(config):
    device = get_device(config)
    print(f"사용 디바이스: {device}")
    
    # 토크나이저 로드
    tokenizer = SimpleTokenizer()
    tokenizer.load(config['tokenizer']['save_path'])
    
    # 모델 로드
    model = load_model(config, device)
    
    print("\n인터랙티브 모드에 오신 것을 환영합니다!")
    print("텍스트를 입력하면 GPT-2가 이어서 생성합니다. 종료하려면 'exit'를 입력하세요.\n")
    
    while True:
        prompt = input("입력: ")
        if prompt.lower() == 'exit':
            print("인터랙티브 모드를 종료합니다.")
            break
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_length=config['model']['max_length'],
            temperature=1.0,
            top_k=50
        )
        print(f"생성된 텍스트:\n{generated_text}\n")

if __name__ == "__main__":
    config = load_config()
    interactive_mode(config)
