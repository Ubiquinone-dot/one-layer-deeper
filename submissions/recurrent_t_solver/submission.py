"""Tied recurrent Transformer for end-to-end repeated modular squaring."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import (
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    TokenLossBatch,
    assert_model_state,
)


D_MODEL = 128
NUM_HEADS = 4
MLP_WIDTH = 384
NUM_LOOPS = 64
TRAINING_LOOPS = 16
TIME_MARKER_TOKEN = 4
DIGIT_OFFSET = 7


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight)


class RecurrentBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.attention_out = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.mlp_norm = RMSNorm(D_MODEL)
        self.mlp_gate = nn.Linear(D_MODEL, MLP_WIDTH, bias=False)
        self.mlp_up = nn.Linear(D_MODEL, MLP_WIDTH, bias=False)
        self.mlp_down = nn.Linear(MLP_WIDTH, D_MODEL, bias=False)

    def forward(self, x: Tensor, attention_mask: Tensor | None) -> Tensor:
        residual = x
        normalized = self.attention_norm(x)
        batch, length, _ = normalized.shape
        q, k, v = self.qkv(normalized).chunk(3, dim=-1)
        head_width = D_MODEL // NUM_HEADS
        q = q.view(batch, length, NUM_HEADS, head_width).transpose(1, 2)
        k = k.view(batch, length, NUM_HEADS, head_width).transpose(1, 2)
        v = v.view(batch, length, NUM_HEADS, head_width).transpose(1, 2)
        mask = None
        if attention_mask is not None:
            if attention_mask.shape == (batch, length):
                mask = attention_mask[:, None, None, :]
            elif attention_mask.shape == (batch, length, length):
                mask = attention_mask[:, None, :, :]
            else:
                raise ValueError("invalid attention_mask shape")
            mask = mask.to(device=x.device, dtype=torch.bool)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attended = attended.transpose(1, 2).contiguous().view(batch, length, D_MODEL)
        x = residual + self.attention_out(attended)
        normalized = self.mlp_norm(x)
        mixed = F.silu(self.mlp_gate(normalized)) * self.mlp_up(normalized)
        return x + self.mlp_down(mixed)


class Model(nn.Module):
    """Depth-controlled sequence model with one shared learned transition."""

    num_loops = NUM_LOOPS

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.input_block = RecurrentBlock()
        self.recurrent_block = RecurrentBlock()
        self.source_norm = RMSNorm(D_MODEL)
        self.source_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @staticmethod
    def _time_steps(input_ids: Tensor) -> Tensor:
        after_marker = input_ids.eq(TIME_MARKER_TOKEN).cumsum(dim=1).gt(0)
        is_digit = input_ids.ge(DIGIT_OFFSET) & input_ids.lt(DIGIT_OFFSET + 10)
        time_steps = torch.zeros(input_ids.shape[0], device=input_ids.device)
        for position in range(input_ids.shape[1]):
            include = after_marker[:, position] & is_digit[:, position]
            digit = (input_ids[:, position] - DIGIT_OFFSET).to(time_steps.dtype)
            time_steps = torch.where(include, 10.0 * time_steps + digit, time_steps)
        return time_steps.clamp(min=1.0, max=float(NUM_LOOPS))

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        source = self.token_embedding(input_ids) + self.position_embedding(positions)
        encoded = self.input_block(source, attention_mask)
        state = encoded
        source_residual = self.source_projection(self.source_norm(encoded))
        time_steps = self._time_steps(input_ids)
        loop_count = TRAINING_LOOPS if self.training else NUM_LOOPS
        for iteration in range(loop_count):
            transformed = self.recurrent_block(state, attention_mask)
            candidate = state + 0.25 * (transformed - state + source_residual)
            active = time_steps.gt(float(iteration))[:, None, None]
            state = torch.where(active, candidate, state)
        return self.head(self.final_norm(state)), None


class DeviceAdamW(torch.optim.Optimizer):
    """AdamW with every optimizer-state tensor on the parameter device."""

    def __init__(
        self,
        parameters,
        *,
        lr: float,
        betas: tuple[float, float],
        weight_decay: float,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(
            parameters,
            {
                "lr": lr,
                "betas": betas,
                "weight_decay": weight_decay,
                "eps": eps,
            },
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            learning_rate = group["lr"]
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                state = self.state[parameter]
                if not state:
                    state["step"] = torch.zeros((), device=parameter.device)
                    state["first_moment"] = torch.zeros_like(
                        parameter,
                        dtype=torch.float32,
                    )
                    state["second_moment"] = torch.zeros_like(
                        parameter,
                        dtype=torch.float32,
                    )
                step = state["step"].add_(1)
                first_moment = state["first_moment"]
                second_moment = state["second_moment"]
                float_gradient = gradient.float()
                first_moment.mul_(beta1).add_(float_gradient, alpha=1.0 - beta1)
                second_moment.mul_(beta2).addcmul_(
                    float_gradient,
                    float_gradient,
                    value=1.0 - beta2,
                )
                parameter.mul_(1.0 - learning_rate * group["weight_decay"])
                first_correction = 1.0 - torch.pow(beta1, step)
                second_correction = 1.0 - torch.pow(beta2, step)
                denominator = (
                    second_moment.sqrt() / second_correction.sqrt()
                ).add_(group["eps"])
                update = first_moment / denominator / first_correction
                parameter.add_(update.to(parameter.dtype), alpha=-learning_rate)
        return loss


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    token_losses = F.cross_entropy(
        batch.logits.transpose(1, 2),
        batch.labels,
        ignore_index=-100,
        reduction="none",
    )
    target_counts = batch.valid_mask.sum(dim=1)
    sequence_losses = (
        (token_losses * batch.valid_mask).sum(dim=1)
        / target_counts.clamp_min(1)
    )
    return sequence_losses[target_counts > 0].mean()


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    del spec
    optimizer = DeviceAdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.02,
    )
    return OptimizerBundle(optimizer)


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    batch_size=256,
    eval_batch_size=512,
)
