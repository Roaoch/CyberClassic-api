import uuid
import torch
import json

import pandas as pd

from src.discriminator import DiscriminatorModel

from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GenerationConfig

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
        self.discriminator = DiscriminatorModel.from_pretrained('Roaoch/CyberClassic-Discriminator')

        self.generation_config = GenerationConfig(
            max_new_tokens=max_length,
            num_beams=6,
            early_stopping=True,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden_state  = self.generator(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)['hidden_states'][-1]
        weights_for_non_padding = attention_mask * torch.arange(start=1, end=last_hidden_state.shape[1] + 1).unsqueeze(0)
        sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
        num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
        return sum_embeddings / num_of_none_padding_tokens

    def generate(self) -> str:
        starts = self.startings['text'].values[np.random.randint(0, len(self.startings), 4)].tolist()
        tokens = self.tokenizer(starts, return_tensors='pt', padding=True, truncation=True)
        generated = self.generator.generate(**tokens, generation_config=self.generation_config)

        input_emb = self.encode(input_ids=generated, attention_mask=torch.full(generated.size(), 1))
        score = self.discriminator(input_emb)
        score = torch.abs(score - 0.889)
        index = int(torch.argmin(score))

        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        return decoded[index]

    def answer(self, promt: str) -> str:
        promt_tokens = self.tokenizer(promt, return_tensors='pt')
        output = self.generator.generate(
            **promt_tokens, 
            generation_config=self.generation_config,
        )

        decoded = self.tokenizer.batch_decode(output)
        return decoded[0]