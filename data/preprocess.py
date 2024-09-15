# data/preprocess.py

import os
from tokenizer.tokenizer import SimpleTokenizer
from utils.config import load_config

def preprocess_data(config):
    raw_data_path = config['data']['raw_path']
    processed_data_path = config['data']['processed_path']
    
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 간단한 전처리: 모든 텍스트를 소문자로 변환
    processed_text = text.lower()
    
    with open(processed_data_path, 'w', encoding='utf-8') as f:
        f.write(processed_text)
    
    print("데이터 전처리 완료.")

def main():
    config = load_config()
    preprocess_data(config)

if __name__ == "__main__":
    main()
