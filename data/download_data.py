# data/download_data.py

import os
from datasets import load_dataset
from tokenizer.tokenizer import GPTTokenizer
from utils.config import load_config

def preprocess_text(text):
    """
    텍스트 전처리 함수.
    필요에 따라 추가적인 정제 작업을 수행할 수 있습니다.
    예: 불필요한 공백 제거, 소문자화 등.
    """
    text = ' '.join(text.split())  # 예: 불필요한 공백 제거
    return text

def download_and_process_data(config):
    dataset_name = config['data']['name']        # HuggingFace 데이터셋 이름
    split = config['data']['split']              # 사용할 데이터셋 분할 (예: 'train', 'validation', 'test')
    data_dir = config['data']['dir']
    raw_path = config["data"]["raw_path"]
    processed_path = config["data"]["processed_path"]
    subset_name = config["data"]["subset_name"]
    
    os.makedirs(data_dir, exist_ok=True)
    
    if not os.path.isfile(processed_path):
        print("HuggingFace 데이터셋 로드 중...")
        try:
            if subset_name:
                dataset = load_dataset(dataset_name, subset_name, split=split)
            else:
                dataset = load_dataset(dataset_name, split=split)
        except Exception as e:
            print(f"데이터셋 로드 실패: {e}")
            return
        print(f"데이터셋 '{dataset_name}' 로드 완료. 총 샘플 수: {len(dataset)}")

        dataset = dataset[:1000]
        print(len(dataset['text']))

        with open(raw_path, 'w', encoding='utf-8') as f:
            f.write(' '.join(map(str, dataset['text'])))
        
        tokenizer = GPTTokenizer(config)
        vocab_size = config['tokenizer']['vocab_size']
        
        tokens = []
        print("데이터 인코딩 중...")
        for idx, text in enumerate(dataset['text']):

            # 빈 텍스트는 건너뜁니다.
            if not text.strip():
                continue  

            if idx % 1000 == 0 and idx > 0:
                print(f"인코딩 진행 중: {idx} / {len(dataset)}")
            text = preprocess_text(text)
            encoded = tokenizer.encode(text)
            
            if len(encoded) == 0:
                continue

            if max(encoded) >= vocab_size:
                raise ValueError(f"인코딩된 토큰 {max(encoded)}이 vocab_size {vocab_size}을 초과합니다.")
            tokens.extend(encoded)
        print("데이터 인코딩 완료.")
        
        print(f"처리된 데이터를 '{processed_path}'에 저장 중...")
        with open(processed_path, 'w', encoding='utf-8') as f:
            f.write(' '.join(map(str, tokens)))  # 토큰을 공백으로 구분하여 저장
        print("처리된 데이터 저장 완료.")
    else:
        print(f"처리된 데이터가 '{processed_path}'에 이미 존재합니다.")

def main():
    config = load_config()
    download_and_process_data(config)

if __name__ == "__main__":
    main()
