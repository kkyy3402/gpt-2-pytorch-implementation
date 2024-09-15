# data/download_data.py

import os
import requests
from utils.config import load_config

def download_data(config):
    data_url = config['data']['url']
    data_dir = config['data']['dir']
    data_path = config['data']['raw_path']
    
    os.makedirs(data_dir, exist_ok=True)
    if not os.path.isfile(data_path):
        print("데이터 다운로드 중...")
        response = requests.get(data_url)
        with open(data_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("데이터 다운로드 완료.")
    else:
        print("데이터가 이미 존재합니다.")

def main():
    config = load_config()
    download_data(config)

if __name__ == "__main__":
    main()
