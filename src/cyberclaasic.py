import uuid
import torch
import json

import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, GPT2LMHeadModel, GenerationConfig

import numpy as np

class CyberClassic(torch.nn.Module):
    def __init__(
            self,
            max_length: int,
            startings_path: str
        ) -> None:
        super().__init__()
        self.max_length = max_length
        self.startings = pd.read_csv(startings_path)

        self.tokenizer = AutoTokenizer.from_pretrained('Roaoch/CyberClassic-Generator')
        self.generator: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained('Roaoch/CyberClassic-Generator')
        self.discriminator_tokenizer = AutoTokenizer.from_pretrained('Roaoch/CyberClassic-Discriminator')
        self.discriminator = AutoModelForSequenceClassification.from_pretrained('Roaoch/CyberClassic-Discriminator')

        self.generation_config = GenerationConfig(
            max_new_tokens=max_length,
            num_beams=6,
            early_stopping=True,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )

    def generate(self) -> str:
        starts = self.startings['text'].values[np.random.randint(0, len(self.startings), 4)].tolist()
        tokens = self.tokenizer(starts, return_tensors='pt', padding=True, truncation=True)
        generated = self.generator.generate(**tokens, generation_config=self.generation_config)

        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        decoded_tokens = self.discriminator_tokenizer(decoded, return_tensors='pt', padding=True, truncation=True)
        score = self.discriminator(**decoded_tokens)
        index = int(torch.argmax(score.logits))

        return decoded[index]

    def answer(self, promt: str) -> str:
        promt = promt + '. '
        length = len(promt)

        promt_tokens = self.tokenizer(promt, return_tensors='pt')
        output = self.generator.generate(
            **promt_tokens, 
            generation_config=self.generation_config,
        )

        decoded = self.tokenizer.batch_decode(output)
        return decoded[0][length:].strip()