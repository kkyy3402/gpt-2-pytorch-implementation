# interactive/interactive.py

import torch
from tokenizer.tokenizer import GPTTokenizer
from model.gpt2_model import GPT2
from utils.device_setup import get_device
from utils.config import load_config

def load_model(config, device, vocab_size):
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

def generate_text(model, tokenizer, device, prompt, max_length=100, temperature=1.0, top_k=50, vocab_size=10000):
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)  # LongTensor로 유지
    
    # 모델을 fp16으로 변환 (입력 텐서 제외)
    model = model.half()  # 모델을 fp16으로 변환

    generated = input_ids
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(generated)  # input_ids는 LongTensor로 유지
            
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                raise ValueError("모델 출력에 NaN 또는 무한대 값이 포함되어 있습니다.")
            
            logits = outputs[:, -1, :] / temperature
            top_k = min(top_k, logits.size(-1))
            
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            
            probabilities = torch.softmax(top_k_logits, dim=-1)
            
            # probabilities 검증
            assert torch.all(probabilities >= 0), "probabilities에 음수 값이 있습니다."
            assert torch.all(probabilities <= 1), "probabilities에 1을 초과하는 값이 있습니다."
            # assert torch.allclose(probabilities.sum(dim=-1), torch.tensor(1.0, device=probabilities.device, dtype=probabilities.dtype)), "probabilities의 합이 1이 아닙니다."
            
            next_token = torch.multinomial(probabilities, num_samples=1)
            next_token = top_k_indices.gather(-1, next_token)
            
            # next_token 검증
            assert torch.max(next_token) < vocab_size, "next_token이 vocab_size를 초과합니다."
            
            generated = torch.cat((generated, next_token), dim=1)
            
            # test_str = tokenizer.decode(generated.squeeze().tolist(), is_batch=False)
            
    generated_text = tokenizer.decode(generated.squeeze().tolist(), is_batch=False)
    return generated_text


def interactive_mode(config):
    device = get_device(config)
    print(f"사용 디바이스: {device}")
    
    # 토크나이저 로드
    tokenizer = GPTTokenizer(config)
    vocab_size = tokenizer.get_vocab_size()

    # 모델 로드
    model = load_model(config, device, vocab_size)
    
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
            top_k=50, 
            vocab_size=vocab_size
        )
        print(f"생성된 텍스트:\n{generated_text}\n")

if __name__ == "__main__":
    config = load_config()
    interactive_mode(config)
