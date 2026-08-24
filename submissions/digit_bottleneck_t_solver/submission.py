"""Tied digit-bottleneck recurrence trained only on final answer tokens."""

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


WIDTH = 64
NUM_HEADS = 4
MLP_WIDTH = 128
NUM_LOOPS = 64
TRAINING_LOOPS = 16
INNER_LOOPS = 1
DIGIT_OFFSET = 7
N_TOKEN = 2
X_TOKEN = 3
T_TOKEN = 4


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


class Mixer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(WIDTH)
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH, bias=False)
        self.attention_out = nn.Linear(WIDTH, WIDTH, bias=False)
        self.mlp_norm = RMSNorm(WIDTH)
        self.up = nn.Linear(WIDTH, 2 * MLP_WIDTH, bias=False)
        self.down = nn.Linear(MLP_WIDTH, WIDTH, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        q, k, v = self.qkv(self.attention_norm(x)).chunk(3, dim=-1)
        head_width = WIDTH // NUM_HEADS
        q = q.view(batch, length, NUM_HEADS, head_width).transpose(1, 2)
        k = k.view(batch, length, NUM_HEADS, head_width).transpose(1, 2)
        v = v.view(batch, length, NUM_HEADS, head_width).transpose(1, 2)
        mixed = F.scaled_dot_product_attention(q, k, v)
        mixed = mixed.transpose(1, 2).contiguous().view(batch, length, WIDTH)
        x = x + self.attention_out(mixed)
        gate, value = self.up(self.mlp_norm(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(gate) * value)


class Model(nn.Module):
    """Re-encodes each recurrent state through ten learned digit symbols."""

    num_loops = NUM_LOOPS

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.num_slots = spec.max_seq_len
        self.digit_embedding = nn.Embedding(10, WIDTH)
        self.position_embedding = nn.Embedding(spec.max_seq_len, WIDTH)
        self.role_embedding = nn.Embedding(2, WIDTH)
        self.mixers = nn.ModuleList(Mixer() for _ in range(INNER_LOOPS))
        self.transition_norm = RMSNorm(WIDTH)
        self.transition_head = nn.Linear(WIDTH, 10, bias=False)
        self.output_norm = RMSNorm(WIDTH)
        self.output_head = nn.Linear(WIDTH, spec.vocab_size, bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.03)

    @staticmethod
    def _time_steps(input_ids: Tensor) -> Tensor:
        after_marker = input_ids.eq(T_TOKEN).cumsum(dim=1).gt(0)
        is_digit = input_ids.ge(DIGIT_OFFSET) & input_ids.lt(DIGIT_OFFSET + 10)
        value = torch.zeros(input_ids.shape[0], device=input_ids.device)
        for position in range(input_ids.shape[1]):
            include = after_marker[:, position] & is_digit[:, position]
            digit = (input_ids[:, position] - DIGIT_OFFSET).clamp(0, 9).float()
            value = torch.where(include, 10.0 * value + digit, value)
        return value.clamp(1, NUM_LOOPS)

    def _digit_tape(
        self,
        input_ids: Tensor,
        start_token: int,
        stop_token: int,
    ) -> Tensor:
        after_start = input_ids.eq(start_token).cumsum(dim=1).gt(0)
        before_stop = input_ids.eq(stop_token).cumsum(dim=1).eq(0)
        is_digit = input_ids.ge(DIGIT_OFFSET) & input_ids.lt(DIGIT_OFFSET + 10)
        include = after_start & before_stop & is_digit
        digits = torch.zeros(
            input_ids.shape[0], self.num_slots, dtype=torch.long, device=input_ids.device
        )
        for position in range(input_ids.shape[1]):
            place = include[:, position + 1 :].sum(dim=1).clamp_max(self.num_slots - 1)
            digit = (input_ids[:, position] - DIGIT_OFFSET).clamp(0, 9)
            current = digits.gather(1, place[:, None]).squeeze(1)
            selected = torch.where(include[:, position], digit, current)
            digits = digits.scatter(1, place[:, None], selected[:, None])
        return self.digit_embedding(digits)

    def _transition(self, state: Tensor, modulus: Tensor) -> Tensor:
        positions = self.position_embedding.weight[None, :, :]
        work = state + positions + self.role_embedding.weight[0]
        context = modulus + positions + self.role_embedding.weight[1]
        tokens = torch.cat((work, context), dim=1)
        for mixer in self.mixers:
            tokens = mixer(tokens)
        candidate = tokens[:, : self.num_slots]
        digit_logits = self.transition_head(self.transition_norm(candidate))
        probabilities = digit_logits.softmax(dim=-1)
        if not self.training:
            hard = F.one_hot(probabilities.argmax(dim=-1), 10).to(probabilities.dtype)
            probabilities = probabilities + (hard - probabilities).detach()
        return probabilities @ self.digit_embedding.weight

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        state = self._digit_tape(input_ids, X_TOKEN, T_TOKEN)
        modulus = self._digit_tape(input_ids, N_TOKEN, X_TOKEN)
        time_steps = self._time_steps(input_ids)
        loop_count = TRAINING_LOOPS if self.training else NUM_LOOPS
        for iteration in range(loop_count):
            candidate = self._transition(state, modulus)
            state = torch.where(time_steps.gt(iteration)[:, None, None], candidate, state)

        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device)[None, :]
        if attention_mask is None:
            valid_lengths = torch.full(
                (batch, 1), length, dtype=torch.long, device=input_ids.device
            )
        else:
            valid_lengths = attention_mask.long().sum(dim=1, keepdim=True)
        reverse_positions = (valid_lengths - 1 - positions).clamp(
            min=0, max=self.num_slots - 1
        )
        batch_indices = torch.arange(batch, device=input_ids.device)[:, None]
        output_state = state[batch_indices, reverse_positions]
        return self.output_head(self.output_norm(output_state)), None


class DeviceAdamW(torch.optim.Optimizer):
    def __init__(self, parameters) -> None:
        super().__init__(
            parameters,
            {"lr": 2e-3, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.01},
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state["step"] = torch.zeros((), device=parameter.device)
                    state["first_moment"] = torch.zeros_like(parameter, dtype=torch.float32)
                    state["second_moment"] = torch.zeros_like(parameter, dtype=torch.float32)
                step = state["step"].add_(1)
                gradient = parameter.grad.float()
                first = state["first_moment"]
                second = state["second_moment"]
                first.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                second.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                first_correction = 1.0 - torch.pow(beta1, step)
                second_correction = 1.0 - torch.pow(beta2, step)
                denominator = (second.sqrt() / second_correction.sqrt()).add_(group["eps"])
                update = first / denominator / first_correction
                parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])
        return loss


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    losses = F.cross_entropy(
        batch.logits.transpose(1, 2), batch.labels, ignore_index=-100, reduction="none"
    )
    counts = batch.valid_mask.sum(dim=1)
    cross_entropy = ((losses * batch.valid_mask).sum(dim=1) / counts.clamp_min(1))[
        counts > 0
    ].mean()
    probabilities = batch.logits.float().softmax(dim=-1)[
        ..., DIGIT_OFFSET : DIGIT_OFFSET + 10
    ]
    digits = torch.arange(10, device=batch.logits.device).float()
    expected_digits = (probabilities * digits).sum(dim=-1)
    target_digits = (batch.labels - DIGIT_OFFSET).clamp(0, 9).float()
    slots = torch.arange(batch.labels.shape[1], device=batch.logits.device)
    exponents = (counts[:, None] - 1 - slots[None, :]).clamp_min(0)
    places = torch.pow(10.0, exponents.float()) * batch.valid_mask
    normalizer = torch.pow(10.0, counts.float()).clamp_min(1.0)
    expected_number = (expected_digits * places).sum(dim=1) / normalizer
    target_number = (target_digits * places).sum(dim=1) / normalizer
    return cross_entropy + F.smooth_l1_loss(expected_number, target_number)


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    del spec
    return OptimizerBundle(DeviceAdamW(model.parameters()))


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    batch_size=256,
    eval_batch_size=512,
)
