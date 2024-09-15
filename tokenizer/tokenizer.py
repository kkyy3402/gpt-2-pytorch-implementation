import os
import tiktoken

class GPTTokenizer:
    def __init__(self, config):
        self.config = config
        # tiktoken에서 기본 BPE 토크나이저를 로드 (GPT-2와 같은 모델을 기준으로 사용)
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def encode(self, text):
        # 텍스트를 토큰 ID로 변환
        return self.tokenizer.encode(text)

    def decode(self, tokens, is_batch=True):
        # 토큰 ID를 텍스트로 변환
        if is_batch:
            return [self.tokenizer.decode(token_ids) for token_ids in tokens]
        else:
            return self.tokenizer.decode(tokens)

    # def load(self, path):
    #     # tiktoken에서 사전 저장된 모델을 로드할 수 없습니다.
    #     print(f"tiktoken은 별도의 모델 파일 로딩을 지원하지 않습니다. GPT-2 기반 토크나이저를 사용 중입니다.")
