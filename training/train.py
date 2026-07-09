"""Train the kilonova transformer on the real Hourglass survey data.

Run on Colab GPU:
    !python train.py --data-dir data/dust_generation --epochs 30

The KN class (index 3) is empty in this survey-only set — the model learns to separate the
three contaminant classes {Ia, II, other}. KN injection arrives with the (not-yet-included)
LANL dataloader; the architecture already reserves the slot.
"""

import argparse
import os

import torch
import torch.nn as nn

from hourglass_data import build_dataloaders, GROUP_ORDER
from model import KilonovaTransformer

MODEL_INPUT_KEYS = [
    'delta_time', 'band_index', 'token_type_index',
    'magnitude', 'sigma_magnitude', 'magnitude_mask', 'sigma_mask',
    'redshift', 'redshift_error', 'has_redshift', 'padding_mask', 'label',
]


def move_batch_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items() if key in MODEL_INPUT_KEYS}


def class_weights_from_loader(train_loader, num_classes, device):
    """Inverse-frequency class weights from the training labels (Ia/II dominate, other rare)."""
    counts = torch.zeros(num_classes)
    for batch in train_loader:
        counts += torch.bincount(batch['label'], minlength=num_classes)
    present = counts > 0
    weights = torch.zeros(num_classes)
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return weights.to(device)


@torch.no_grad()
def evaluate(model, loader, loss_function, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        logits = model(batch)
        loss = loss_function(logits, batch['label'])
        total_loss += loss.item() * batch['label'].shape[0]
        total_correct += int((logits.argmax(dim=-1) == batch['label']).sum())
        total_examples += batch['label'].shape[0]
    return total_loss / total_examples, total_correct / total_examples


def train(data_dir, epochs, batch_size, learning_rate, weight_decay,
          checkpoint_path, num_workers, seed):
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    data = build_dataloaders(
        objects_path=os.path.join(data_dir, 'hourglass_objects.parquet'),
        photometry_path=os.path.join(data_dir, 'hourglass_photometry.parquet'),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    train_loader = data['train_loader']
    validation_loader = data['validation_loader']
    print(f"split sizes: {data['split_sizes']}")
    print(f"normalization: {data['normalization']}")

    model = KilonovaTransformer().to(device)
    number_of_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f'model parameters: {number_of_parameters:,}')

    weights = class_weights_from_loader(train_loader, num_classes=len(GROUP_ORDER), device=device)
    print(f'class weights {GROUP_ORDER}: {weights.tolist()}')
    loss_function = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_validation_accuracy = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_examples = 0
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = loss_function(logits, batch['label'])
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch['label'].shape[0]
            running_examples += batch['label'].shape[0]

        train_loss = running_loss / running_examples
        validation_loss, validation_accuracy = evaluate(model, validation_loader, loss_function, device)
        marker = ''
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'normalization': data['normalization'],
                    'epoch': epoch,
                    'validation_accuracy': validation_accuracy,
                },
                checkpoint_path,
            )
            marker = '  <- best (saved)'
        print(f'epoch {epoch:3d}  train_loss {train_loss:.4f}  '
              f'val_loss {validation_loss:.4f}  val_acc {validation_accuracy:.4f}{marker}')

    print(f'best validation accuracy: {best_validation_accuracy:.4f}  ->  {checkpoint_path}')


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data/dust_generation',
                        help='directory holding hourglass_objects.parquet and hourglass_photometry.parquet')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--checkpoint-path', default='kilonova_transformer.pt')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_arguments()
    train(
        data_dir=arguments.data_dir,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        checkpoint_path=arguments.checkpoint_path,
        num_workers=arguments.num_workers,
        seed=arguments.seed,
    )
