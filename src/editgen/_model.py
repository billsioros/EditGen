from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import AutoProcessor, MusicgenForConditionalGeneration

from editgen._base_controller import EmptyController, BaseController
from editgen._attention import register_attention_control


class ModelProxy(object):
    def __init__(self, model_name: str, guidance_scale: float = 3.0):
        self.model_name = model_name
        self.guidance_scale = guidance_scale

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = MusicgenForConditionalGeneration.from_pretrained(model_name).to(
            self.device
        )

    def generate(
        self, inputs: dict[str, Any], max_new_tokens: int = 512
    ) -> NDArray[np.float_]:
        return (
            self.model.generate(
                **inputs.to(self.device),
                do_sample=True,
                guidance_scale=self.guidance_scale,
                max_new_tokens=max_new_tokens,
            )
            .cpu()
            .numpy()
            .squeeze()
        )

    def encode(self, prompts: list[str]) -> dict[str, Any]:
        return self.processor(text=prompts, padding=True, return_tensors="pt")

    def decode(self, encoded_token: NDArray[np.float_]) -> str:
        return self.processor.decode(encoded_token)

    @property
    def sampling_rate(self) -> float:
        return self.model.config.audio_encoder.sampling_rate

    @property
    def frame_rate(self) -> float:
        return self.model.config.audio_encoder.frame_rate

    @property
    def decoder_layers(self) -> list[torch.nn.Module]:
        return self.model.decoder.model.decoder.layers


class EditGenPipeline(ModelProxy):
    def __init__(
        self,
        model_name: str,
        guidance_scale: float = 3.0,
        seed: int = 0,
        audio_length: float = 10.0,
    ):
        super().__init__(model_name, guidance_scale)

        self._seed = seed
        self._audio_length = audio_length

    def __call__(
        self,
        controller: BaseController,
        *prompts: list[str],
    ) -> NDArray[np.float_]:
        max_new_tokens = 2 ** round(np.log2(self.audio_length * self.frame_rate))

        if self._seed is not None:
            torch.manual_seed(self._seed)

        if controller is None:
            controller = EmptyController()

        controller.reset()
        controller.batch_size = len(prompts)
        controller.max_new_tokens = max_new_tokens

        register_attention_control(self, controller)

        inputs = self.encode(prompts)
        audio_values = self.generate(inputs, max_new_tokens=max_new_tokens)

        return audio_values
