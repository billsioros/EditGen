import torch

from editgen._base_controller import BaseController
from editgen._tokens import get_ignore_indices
from editgen._model import ModelProxy


class BaseEditController(BaseController):
    def replace_self_attention(self, attn):
        attn_base, att_replace = attn[0], attn[1:]

        return attn_base.unsqueeze(0).expand(att_replace.shape[0] + 1, *attn_base.shape)


class RandomController(BaseEditController):
    def replace_cross_attention(self, attn_weights, is_cross, attention_type):
        attn_weights[1:] = torch.randn_like(attn_weights[0])

        return attn_weights


class IgnoreWordController(BaseEditController):
    def __init__(self, indices: list[int]):
        super().__init__()

        self.indices = indices

    def replace_cross_attention(self, attn_weights, is_cross, attention_type):
        attn_weights[1:, :, :, self.indices] = 0

        return attn_weights

    @classmethod
    def from_prompts(
        cls,
        model: ModelProxy,
        prompts: list[str, str],
    ) -> tuple[list[str, str], "IgnoreWordController"]:
        prompts, indices = get_ignore_indices(model, prompts)

        return prompts, IgnoreWordController(indices)


class ReplaceWordController(BaseEditController):
    def __init__(self, indices: list[list[int]], blend: float = 0.5):
        super().__init__()

        self.source_indices, self.target_indices = indices
        self.blend = blend

    def replace_cross_attention(self, attn_weights, is_cross, attention_type):
        source_attn = attn_weights[0, :, :, self.source_indices]
        averaged_attn = source_attn.mean(dim=-1, keepdims=True)

        # Repeat averaged attention values to match dimensions of the target attention
        averaged_attn_repeated = averaged_attn.expand(-1, -1, len(self.target_indices))

        attn_weights[1:, :, :, self.target_indices] = (1 - self.blend) * attn_weights[
            1:, :, :, self.target_indices
        ] + self.blend * averaged_attn_repeated

        return attn_weights


class RefineController(BaseEditController):
    def __init__(self, indices: list[list[int]], blend: float = 0.5):
        super().__init__()

        self.source_indices, self.target_indices = indices
        self.blend = blend

    def replace_cross_attention(self, attn_weights, is_cross, attention_type):
        attn_weights[1:, :, :, self.target_indices] = (1 - self.blend) * attn_weights[
            1:, :, :, self.target_indices
        ] + self.blend * attn_weights[0, :, :, self.source_indices]

        return attn_weights


class ReweightWordController(BaseEditController):
    def __init__(self, indices: list[int], weight: float = 5):
        super().__init__()

        self.indices = indices
        self.weight = weight

    def replace_cross_attention(self, attn_weights, is_cross, attention_type):
        attn_weights[1:] = attn_weights[0]

        non_target_indices = [
            i for i in range(attn_weights.shape[-1]) if i not in self.indices
        ]
        attn_weights[1:, :, :, non_target_indices] /= self.weight

        return attn_weights


class ReplaceController(BaseEditController):
    def __init__(self, blend: float = 0.5):
        super().__init__()

        self.blend = blend

    def replace_cross_attention(self, attn_weights, is_cross, attention_type):
        attn_weights[1:] = (1 - self.blend) * attn_weights[
            1:
        ] + self.blend * attn_weights[0]

        return attn_weights
