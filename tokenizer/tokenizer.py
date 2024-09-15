import os
import tiktoken

class GPTTokenizer:
    def __init__(self, config):
        self.config = config
        # tiktoken에서 기본 BPE 토크나이저를 로드 (GPT-2와 같은 모델을 기준으로 사용)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")

    def encode(self, text):
        # 텍스트를 토큰 ID로 변환
        return self.tokenizer.encode(text)

    def decode(self, tokens, is_batch=True):
        # 토큰 ID를 텍스트로 변환
        if is_batch:
            return [self.tokenizer.decode(token_ids) for token_ids in tokens]
        else:
            return self.tokenizer.decode(tokens)

    def get_vocab_size(self):
        return self.tokenizer.n_vocab
