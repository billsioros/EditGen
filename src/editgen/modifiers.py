from typing import Any

from editgen._base_controller import BaseController


class ControllerModifier(BaseController):
    def __init__(self, controller: BaseController) -> None:
        super().__init__()

        self.controller = controller

    def __getattr__(self, name: str):
        if name == "controller":
            return super().__getattr__(name)

        return getattr(self.controller, name)

    def __setattr__(self, name: str, value: Any):
        if name == "controller":
            return super().__setattr__(name, value)

        return setattr(self.controller, name, value)


class OffsetControllerModifier(ControllerModifier):
    def __init__(self, controller: BaseController, offset: float = 0.0) -> None:
        super().__init__(controller)

        assert 0.0 <= offset <= 1.0

        self.offset = offset

    def replace_self_attention(self, attn):
        if self.cur_step < round(self.offset * self.max_new_tokens):
            return attn

        return self.controller.replace_self_attention(attn)

    def replace_cross_attention(self, attn_weights, is_cross, attention_type):
        if self.cur_step < round(self.offset * self.max_new_tokens):
            return attn_weights

        return self.controller.replace_cross_attention(
            attn_weights, is_cross, attention_type
        )


class AttentionHeadControllerModifier(ControllerModifier):
    def __init__(
        self, controller: BaseController, attention_head_indices: list[int]
    ) -> None:
        super().__init__(controller)

        self.attention_head_indices = attention_head_indices

    def replace_self_attention(self, attn):
        return self.controller.replace_self_attention(attn)

    def replace_cross_attention(self, attn_weights, is_cross, attention_type):
        attn_weights_slice = attn_weights[:, self.attention_head_indices, :, :]
        attn_weights_slice = self.controller.replace_cross_attention(
            attn_weights_slice, is_cross, attention_type
        )

        attn_weights[:, self.attention_head_indices, :, :] = attn_weights_slice

        return attn_weights


class SelfAttentionLerpControllerModifier(ControllerModifier):
    def __init__(self, controller: BaseController) -> None:
        super().__init__(controller)

    def replace_self_attention(self, attn):
        blend = self.cur_att_layer / self.num_att_layers

        attn[1:] = (1 - blend) * attn[1:] + blend * attn[0]

        return attn

    def replace_cross_attention(self, attn_weights, is_cross, attention_type) -> None:
        return self.controller.replace_cross_attention(
            attn_weights, is_cross, attention_type
        )


class SelfAttentionCutoffControllerModifier(ControllerModifier):
    def __init__(self, controller: BaseController, threshold: float = 0.75) -> None:
        super().__init__(controller)

        assert 0.0 <= threshold <= 1.0

        self.threshold = threshold

    def replace_self_attention(self, attn):
        if self.cur_att_layer <= np.floor(self.threshold * self.num_att_layers):
            return self.controller.replace_self_attention(attn)

        return attn

    def replace_cross_attention(self, attn_weights, is_cross, attention_type) -> None:
        return self.controller.replace_cross_attention(
            attn_weights, is_cross, attention_type
        )


class AttentionLerpControllerModifier(ControllerModifier):
    def __init__(self, controller: BaseController) -> None:
        super().__init__(controller)

    def replace_self_attention(self, attn):
        blend = self.cur_att_layer / self.num_att_layers

        attn = blend * attn + (1 - blend) * self.controller.replace_self_attention(attn)

        return attn

    def replace_cross_attention(self, attn_weights, is_cross, attention_type) -> None:
        blend = self.cur_att_layer / self.num_att_layers

        attn_weights = blend * attn_weights + (
            1 - blend
        ) * self.controller.replace_cross_attention(
            attn_weights.clone(), is_cross, attention_type
        )

        return attn_weights


class AttentionCutoffControllerModifier(ControllerModifier):
    def __init__(self, controller: BaseController, threshold: float = 0.75) -> None:
        super().__init__(controller)

        assert 0.0 <= threshold <= 1.0

        self.threshold = threshold

    def replace_self_attention(self, attn):
        if self.cur_att_layer <= np.floor(self.threshold * self.num_att_layers):
            return self.controller.replace_self_attention(attn)

        return attn

    def replace_cross_attention(self, attn_weights, is_cross, attention_type) -> None:
        if self.cur_att_layer <= np.floor(self.threshold * self.num_att_layers):
            return self.controller.replace_cross_attention(
                attn_weights, is_cross, attention_type
            )

        return attn_weights


class DecoderLayerControllerModifier(ControllerModifier):
    def __init__(
        self, controller: BaseController, decoder_layer_indices: set[int]
    ) -> None:
        super().__init__(controller)

        self.decoder_layer_indices = decoder_layer_indices

    def replace_self_attention(self, attn):
        if self.cur_att_layer in self.decoder_layer_indices:
            return self.controller.replace_self_attention(attn)

        return attn

    def replace_cross_attention(self, attn_weights, is_cross, attention_type):
        if self.cur_att_layer in self.decoder_layer_indices:
            return self.controller.replace_cross_attention(
                attn_weights, is_cross, attention_type
            )

        return attn_weights
