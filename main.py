# main.py

import sys
import os
import argparse
from utils.config import load_config
from data.download_data import download_and_process_data
from data.preprocess import preprocess_data
from tokenizer.tokenizer import GPTTokenizer
from training.train import train_model
from interactive.interactive import interactive_mode

def main():
    # 프로젝트 루트 디렉토리를 PYTHONPATH에 추가
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # argparse를 사용하여 명령어 옵션 처리
    parser = argparse.ArgumentParser(description="GPT-2 Custom Pretrain 프로젝트")
    parser.add_argument('--interactive', action='store_true', help="인터랙티브 모드로 전환")
    args = parser.parse_args()

    print(args)
    
    # 설정 로드
    config = load_config()
    
    if args.interactive:
        # 인터랙티브 모드로 전환
        interactive_mode(config)
    else:
        # 전체 파이프라인 실행
        print("1단계: 데이터 다운로드...")
        download_and_process_data(config)

        print("2단계: 모델 학습...")
        train_model(config)

        print("모든 단계가 성공적으로 완료되었습니다.")

if __name__ == "__main__":
    main()
