import torch

from transformers import PretrainedConfig, PreTrainedModel

class DiscriminatorModelConfig(PretrainedConfig):
    model_type = 'descriminatormodel'
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class DiscriminatorModel(PreTrainedModel):
    config_class = DiscriminatorModelConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 1),
            torch.nn.Dropout(0.1),
            torch.nn.Sigmoid()
        )
    def forward(self, input):
        return self.model(input) 