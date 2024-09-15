# tokenizer/tokenizer.py

import os
import json
from collections import defaultdict
from utils.config import load_config

class SimpleTokenizer:
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
    
    def build_vocab(self, text, vocab_size=10000):
        word_freq = defaultdict(int)
        for word in text.split():
            word_freq[word] += 1
        
        # 빈도순으로 정렬하여 상위 vocab_size 개 단어 선택
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        sorted_words = sorted_words[:vocab_size-2]  # 특별 토큰을 위해 2개 남겨둠
        
        # 특별 토큰 추가
        self.token_to_id["<PAD>"] = 0
        self.token_to_id["<UNK>"] = 1
        self.id_to_token[0] = "<PAD>"
        self.id_to_token[1] = "<UNK>"
        
        for idx, (word, _) in enumerate(sorted_words, start=2):
            self.token_to_id[word] = idx
            self.id_to_token[idx] = word
    
    def encode(self, text):
        result = [self.token_to_id.get(word, self.token_to_id["<UNK>"]) for word in text.split()]
        return result
    
    def decode(self, tokens):
        decoded_texts = [' '.join([self.id_to_token.get(token.item() , "<UNK>") for token in sublist]) for sublist in tokens]
        return decoded_texts
    

    
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "token_to_id.json"), 'w', encoding='utf-8') as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=4)
        with open(os.path.join(path, "id_to_token.json"), 'w', encoding='utf-8') as f:
            json.dump(self.id_to_token, f, ensure_ascii=False, indent=4)
        print(f"토크나이저가 {path}에 저장되었습니다.")
    
    def load(self, path):
        with open(os.path.join(path, "token_to_id.json"), 'r', encoding='utf-8') as f:
            self.token_to_id = json.load(f)
        with open(os.path.join(path, "id_to_token.json"), 'r', encoding='utf-8') as f:
            id_to_token_str_keys = json.load(f)
            # 키를 정수로 변환
            self.id_to_token = {int(k): v for k, v in id_to_token_str_keys.items()}
        print(f"토크나이저가 {path}에서 로드되었습니다.")

def train_tokenizer(config):
    tokenizer = SimpleTokenizer()
    processed_data_path = config['data']['processed_path']
    tokenizer_save_path = config['tokenizer']['save_path']
    
    with open(processed_data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer.build_vocab(text, vocab_size=config['tokenizer']['vocab_size'])
    tokenizer.save(tokenizer_save_path)
