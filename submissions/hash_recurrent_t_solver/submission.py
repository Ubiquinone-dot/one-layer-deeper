"""Learned hash-state recurrence with final-answer supervision only."""

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


WIDTH = 128
HIDDEN_WIDTH = 512
HASH_SIZE = 16_384
MODULUS_HASH_SIZE = 4_096
NUM_LOOPS = 64
TRAINING_LOOPS = 16
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


class Model(nn.Module):
    """A finite learned state space evolved by one shared neural transition."""

    num_loops = NUM_LOOPS

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.x_memory = nn.Embedding(HASH_SIZE, WIDTH)
        self.modulus_memory = nn.Embedding(MODULUS_HASH_SIZE, WIDTH)
        self.token_embedding = nn.Embedding(spec.vocab_size, WIDTH)
        self.reverse_position_embedding = nn.Embedding(spec.max_seq_len, WIDTH)
        self.transition_norm = RMSNorm(WIDTH)
        self.condition_projection = nn.Linear(WIDTH, WIDTH, bias=False)
        self.transition_up = nn.Linear(WIDTH, HIDDEN_WIDTH, bias=False)
        self.transition_down = nn.Linear(HIDDEN_WIDTH, WIDTH, bias=False)
        self.output_norm = RMSNorm(WIDTH)
        self.output_mixer = nn.Linear(WIDTH, WIDTH, bias=False)
        self.head = nn.Linear(WIDTH, spec.vocab_size, bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @staticmethod
    def _number_between(input_ids: Tensor, start_token: int, stop_token: int | None) -> Tensor:
        after_start = input_ids.eq(start_token).cumsum(dim=1).gt(0)
        before_stop = torch.ones_like(after_start)
        if stop_token is not None:
            before_stop = input_ids.eq(stop_token).cumsum(dim=1).eq(0)
        is_digit = input_ids.ge(DIGIT_OFFSET) & input_ids.lt(DIGIT_OFFSET + 10)
        include = after_start & before_stop & is_digit
        value = torch.zeros(
            input_ids.shape[0],
            dtype=torch.long,
            device=input_ids.device,
        )
        for position in range(input_ids.shape[1]):
            digit = (input_ids[:, position] - DIGIT_OFFSET).clamp(min=0, max=9)
            value = torch.where(include[:, position], 10 * value + digit, value)
        return value

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        modulus = self._number_between(input_ids, N_TOKEN, X_TOKEN)
        x_value = self._number_between(input_ids, X_TOKEN, T_TOKEN)
        time_steps = self._number_between(input_ids, T_TOKEN, None).clamp(
            min=1,
            max=NUM_LOOPS,
        )
        condition = self.modulus_memory(modulus.remainder(MODULUS_HASH_SIZE))
        state = self.x_memory(x_value.remainder(HASH_SIZE)) + condition
        projected_condition = self.condition_projection(condition)
        loop_count = TRAINING_LOOPS if self.training else NUM_LOOPS
        for iteration in range(loop_count):
            transition_input = self.transition_norm(state) + projected_condition
            delta = self.transition_down(F.silu(self.transition_up(transition_input)))
            candidate = state + 0.25 * delta
            state = torch.where(time_steps.gt(iteration)[:, None], candidate, state)

        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device)[None, :]
        if attention_mask is None:
            valid_lengths = torch.full(
                (batch, 1),
                length,
                device=input_ids.device,
                dtype=torch.long,
            )
        else:
            valid_lengths = attention_mask.long().sum(dim=1, keepdim=True)
        reverse_positions = (valid_lengths - 1 - positions).clamp(
            min=0,
            max=self.config.max_seq_len - 1,
        )
        output = (
            state[:, None, :]
            + self.reverse_position_embedding(reverse_positions)
            + 0.1 * self.token_embedding(input_ids)
        )
        output = self.output_mixer(self.output_norm(output))
        return self.head(F.silu(output)), None


class DeviceAdamW(torch.optim.Optimizer):
    def __init__(self, parameters) -> None:
        super().__init__(parameters, {"lr": 3e-3, "betas": (0.9, 0.95), "eps": 1e-8})

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
                parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])
        return loss


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    token_losses = F.cross_entropy(
        batch.logits.transpose(1, 2),
        batch.labels,
        ignore_index=-100,
        reduction="none",
    )
    counts = batch.valid_mask.sum(dim=1)
    sequence_loss = (
        (token_losses * batch.valid_mask).sum(dim=1) / counts.clamp_min(1)
    )[counts > 0].mean()

    probabilities = batch.logits.float().softmax(dim=-1)[
        ..., DIGIT_OFFSET : DIGIT_OFFSET + 10
    ]
    digits = torch.arange(10, device=batch.logits.device).float()
    expected_digits = (probabilities * digits).sum(dim=-1)
    target_digits = (batch.labels - DIGIT_OFFSET).clamp(min=0, max=9).float()
    slots = torch.arange(batch.labels.shape[1], device=batch.logits.device)
    exponents = (counts[:, None] - 1 - slots[None, :]).clamp_min(0)
    places = torch.pow(10.0, exponents.float()) * batch.valid_mask
    normalizer = torch.pow(10.0, counts.float()).clamp_min(1.0)
    expected_number = (expected_digits * places).sum(dim=1) / normalizer
    target_number = (target_digits * places).sum(dim=1) / normalizer
    return sequence_loss + F.smooth_l1_loss(expected_number, target_number)


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
