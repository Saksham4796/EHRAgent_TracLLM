import os
import time

import tiktoken
from openai import OpenAI

from .Model import Model
class GPT(Model):
    def __init__(self, config):
        super().__init__(config)
        raw_keys = config["api_key_info"]["api_keys"]
        api_keys = []
        for key in raw_keys:
            if not key:
                continue
            if isinstance(key, str) and key.startswith("${") and key.endswith("}"):
                env_name = key[2:-1]
                env_val = os.getenv(env_name)
                if env_val:
                    api_keys.append(env_val)
            else:
                api_keys.append(key)
        if not api_keys:
            raise ValueError("No API key provided. Set OPENAI_API_KEY in .env or config.")
        api_pos = int(config["api_key_info"]["api_key_use"])
        assert (0 <= api_pos < len(api_keys)), "Please enter a valid API key to use"
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.client = OpenAI(api_key=api_keys[api_pos])
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.seed = 10

    def query(self, msg, max_tokens=128000):
        super().query(max_tokens)
        while True:
            try:
                completion = self.client.chat.completions.create(
                    model=self.name,
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                    seed = self.seed,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": msg}
                    ],
                )
                response = completion.choices[0].message.content
                time.sleep(1)
                break
            except Exception as e:
                print(e)
                time.sleep(10)
        return response
    
    def get_prompt_length(self,msg):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        num_tokens = len(encoding.encode(msg))
        return num_tokens
    
    def cut_context(self,msg,max_length):
        tokens = self.encoding.encode(msg)
        truncated_tokens = tokens[:max_length]
        truncated_text = self.encoding.decode(truncated_tokens)
        return truncated_text
