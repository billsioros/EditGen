from abc import ABC, abstractmethod
from collections import defaultdict

import torch


class BaseController(ABC):
    def reset(self):
        self.num_att_layers = 0
        self.batch_size = -1
        self.max_new_tokens = -1

        self.cur_att_layer = 0
        self.cur_step = 0

    def __call__(self, attn_weights, is_cross, attention_type) -> None:
        self.cur_att_layer = self.cur_att_layer + 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step = self.cur_step + 1

        # Exclude unconditional inputs
        h1 = attn_weights.shape[0] // 2
        attn = attn_weights[:h1]

        # Reshape according to batch size
        h2 = attn.shape[0] // (self.batch_size)

        attn = attn.reshape(self.batch_size, h2, *attn.shape[1:])

        if is_cross:
            attn = self.replace_cross_attention(attn, is_cross, attention_type)
            # attn /= torch.sum(attn, dim=-1, keepdim=True)
        else:
            attn = self.replace_self_attention(attn)

        attn = attn.reshape(self.batch_size * h2, *attn.shape[2:])

        attn_weights[:h1] = attn

        return attn_weights

    @abstractmethod
    def replace_self_attention(self, attn):
        raise NotImplementedError

    @abstractmethod
    def replace_cross_attention(self, attn_weights, is_cross, attention_type) -> None:
        raise NotImplementedError


class EmptyController(BaseController):
    def replace_self_attention(self, attn):
        return attn

    def replace_cross_attention(self, attn_weights, is_cross, attention_type):
        return attn_weights


class AttentionStore(BaseController):
    def reset(self):
        super().reset()

        self.features = defaultdict(lambda: [])

    def get_self_attention(self):
        tensors = self.features["self"]
        tensors = [tensor.mean(dim=-1) for tensor in tensors]
        tensors = torch.stack(tensors)
        tensors = tensors.view(self.max_new_tokens, -1, *tensors.shape[1:])

        return tensors

    def get_cross_attention(self):
        tensors = self.features["cross"]
        tensors = torch.stack(tensors)
        tensors = tensors.view(self.max_new_tokens, -1, *tensors[0].shape)

        return tensors

    def get_self_attention_importance(self):
        aggregate_cross_attention = self.get_self_attention()
        aggregate_cross_attention = aggregate_cross_attention[:, :, 1:, :, :]
        aggregate_cross_attention = aggregate_cross_attention.mean(dim=(0, 3, 4))

        # Min-Max scaling to normalize values between 0 and 1 for each column (sample)
        min_values = aggregate_cross_attention.min(dim=0).values
        max_values = aggregate_cross_attention.max(dim=0).values

        normalized_scores = (aggregate_cross_attention - min_values) / (
            max_values - min_values
        )

        # Get indices that would sort the layers based on their mean scores
        sorted_indices = torch.argsort(normalized_scores, descending=True, dim=0)
        sorted_indices = sorted_indices.view(sorted_indices.shape[1], -1)

        # self-attention layers are called first and thusly hold indices 1, 3, 5 etc.
        sorted_indices = 2 * sorted_indices + 1

        return sorted_indices

    def get_cross_attention_importance(self, word_piece_index):
        aggregate_cross_attention = self.get_cross_attention()
        aggregate_cross_attention = aggregate_cross_attention[
            :, :, 1:, :, :, word_piece_index
        ]
        aggregate_cross_attention = aggregate_cross_attention.mean(dim=(0, 3, 4))

        # Min-Max scaling to normalize values between 0 and 1 for each column (sample)
        min_values = aggregate_cross_attention.min(dim=0).values
        max_values = aggregate_cross_attention.max(dim=0).values

        normalized_scores = (aggregate_cross_attention - min_values) / (
            max_values - min_values
        )

        # Get indices that would sort the layers based on their mean scores
        sorted_indices = torch.argsort(normalized_scores, descending=True, dim=0)
        sorted_indices = sorted_indices.view(sorted_indices.shape[1], -1)

        # cross-attention layers are called second and thusly hold indices 2, 4, 6 etc.
        sorted_indices = 2 * (sorted_indices + 1)

        return sorted_indices

    def get_aggregate_cross_attention(self):
        return torch.mean(torch.stack(self.features["cross"]), axis=0)

    def replace_self_attention(self, attn) -> None:
        self.features["self"].append(attn)

        return attn

    def replace_cross_attention(self, attn_weights, is_cross, attention_type) -> None:
        self.features["cross"].append(attn_weights)

        return attn_weights
