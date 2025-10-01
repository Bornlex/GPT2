import argparse
from dataclasses import dataclass
import numpy as np
import os
import torch
from torch import nn

import config
from src.gpt_config import GPTConfig
from src.model import GPT


@dataclass
class TrainingConfig:
    lr: float
    lr_decay_iters: int
    min_lr: float
    max_iters: int
    beta1: float
    beta2: float
    eval_interval: int
    eval_iters: int
    log_interval: int
    weight_decay: float


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GPT model on Shakespeare dataset')
    parser.add_argument('--save', action='store_true', help='Flag to save the trained model.')
    parser.add_argument('--resume', action='store_true', help='Whether to resume training from checkpoint.')
    parser.add_argument('--save_path', type=str, default='model.pth', help='Path to save the trained model.')
    parser.add_argument('--resume_path', type=str, default='checkpoint.pth', help='Path to save the checkpointed model.')
    args = parser.parse_args()

    return args


def get_batch(batch_size: int, block_size: int, device: str, split: str = 'train'):
    root_directory = os.path.abspath(os.getcwd())
    data_directory = os.path.join(root_directory, 'shakespeare')

    if split == 'train':
        data = np.memmap(os.path.join(data_directory, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_directory, 'val.bin'), dtype=np.uint16, mode='r')

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])

    return x, y


@torch.no_grad()
def estimate_loss(gpt_model, loss_fn, get_batch_fn, device, block_size, eval_iters, batch_size):
    gpt_model.eval()
    losses = []

    for _ in range(eval_iters):
        x, y = get_batch_fn(batch_size, block_size, device, 'val')
        x = x.to(device)
        y = y.to(device, dtype=torch.long)
        logits = gpt_model(x)
        B, T, V = logits.shape
        loss = loss_fn(logits.view(B * T, V), y.view(B * T))
        losses.append(loss.item())

    gpt_model.train()

    return sum(losses) / len(losses)


def train(gpt_model: nn.Module, training_config: TrainingConfig, batch_size: int, block_size: int, checkpoint_path: str = None):
    device = 'mps' if torch.mps.is_available() else 'cpu'

    optimizer = torch.optim.AdamW(
        gpt_model.parameters(),
        lr=training_config.lr,
        betas=(training_config.beta1, training_config.beta2),
        weight_decay=training_config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_config.lr_decay_iters,
        eta_min=training_config.min_lr
    )
    loss_fn = nn.CrossEntropyLoss()

    gpt_model.to(device)
    gpt_model.train()

    best_val_loss = float('inf')
    no_improve_count = 0
    patience = 5

    for iteration in range(training_config.max_iters):
        x, y = get_batch(batch_size, block_size, device, 'train')
        x = x.to(device)
        # y = torch.nn.functional.one_hot(y, gpt_model.config.vocab_size).to(device=device, dtype=torch.float32)
        y = y.to(device, dtype=torch.long)

        prediction = gpt_model(x)
        b, n, c = prediction.shape

        loss = loss_fn(prediction.view(b * n, c), y.view(b * n))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gpt_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if iteration % 10 == 0:
            val_loss = estimate_loss(
                gpt_model, loss_fn, get_batch, device,
                block_size, training_config.eval_iters, batch_size
            )

            input_sentence = 'FIRST CITIZEN:\n'
            gpt_model.eval()
            with torch.no_grad():
                generation = gpt_model.generate(input_sentence, max_tokens=125, temperature=0.9)
            gpt_model.train()
            print(f'[{iteration}|{training_config.max_iters}] loss : {loss.item():.4f} -- val_loss : {val_loss:.4f} -- {generation}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                print(f'Early stopping at iteration {iteration}')
                break

            if checkpoint_path is not None:
                gpt_model.save_weights(checkpoint_path)

    return gpt_model


if __name__ == '__main__':
    arguments = parse_args()

    conf = TrainingConfig(
        lr=config.learning_rate,
        lr_decay_iters=config.lr_decay_iters,
        min_lr=config.min_lr,
        max_iters=config.max_iters,
        beta1=config.beta1,
        beta2=config.beta2,
        eval_interval=config.eval_interval,
        eval_iters=config.eval_iters,
        log_interval=config.log_interval,
        weight_decay=config.weight_decay
    )
    model_config = GPTConfig(
        block_size=config.block_size,
        n_layer=config.n_layers,
        n_head=config.n_head,
        n_embd=config.n_embd,
        vocab_size=config.vocab_size,
        dropout=config.dropout,
        ffn_hidden_size=config.n_embd * 4
    )
    model = GPT(model_config)

    if arguments.resume and os.path.isfile(arguments.resume_path):
        model.load_state_dict(torch.load(arguments.resume_path))

    model = train(
        model,
        conf,
        config.batch_size,
        config.block_size,
        arguments.resume_path
    )

    if arguments.save:
        model.save_weights(arguments.save_path)
