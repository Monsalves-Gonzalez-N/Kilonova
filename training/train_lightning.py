"""Train the kilonova transformer with PyTorch Lightning on the real Hourglass survey data.

Run locally on GPU:
    python train_lightning.py --data-dir data/dust_generation --epochs 30

Lightning port of train.py: same model, loss (inverse-frequency class weights), optimizer
(AdamW) and metrics (loss + accuracy). Methodological binary test: the model separates the
two well-sampled classes {Ia, CCSN}; the Ia-peculiars and rare exotics are dropped, and the
KN injected-signal scenario comes later.
"""

import argparse
import os

import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy

from openuniverse_data import build_dataloaders, GROUP_ORDER
from model import KilonovaTransformer

MODEL_INPUT_KEYS = [
    'delta_time', 'band_index', 'token_type_index',
    'magnitude', 'sigma_magnitude', 'magnitude_mask', 'sigma_mask',
    'redshift', 'redshift_error', 'has_redshift', 'padding_mask', 'label',
]


def class_weights_from_loader(train_loader, num_classes, mode='inverse'):
    """Class weights from the training labels (Ia/II dominate, other rare).

    mode='inverse'  -> full inverse-frequency (aggressive; pushes the rare `other` hard).
    mode='sqrt'     -> sqrt of inverse-frequency (softer; recovers II without dropping `other`).
    mode='none'     -> uniform weights (plain accuracy objective).
    """
    counts = torch.zeros(num_classes)
    for batch in train_loader:
        counts += torch.bincount(batch['label'], minlength=num_classes)
    present = counts > 0
    weights = torch.zeros(num_classes)
    if mode == 'none':
        weights[present] = 1.0
        return weights
    inverse = counts[present].sum() / (present.sum() * counts[present])
    if mode == 'sqrt':
        inverse = inverse.sqrt()
    weights[present] = inverse
    return weights


class LitKilonova(L.LightningModule):
    """LightningModule wrapping KilonovaTransformer with weighted cross-entropy."""

    def __init__(self, class_weights, learning_rate=1e-3, weight_decay=1e-4,
                 num_classes=len(GROUP_ORDER), d_model=128, num_heads=4, num_layers=4,
                 d_feedforward=512, dropout=0.1, max_epochs=30, min_learning_rate=1e-5,
                 warmup_epochs=5, val_regime_names=('3ep_z',)):
        super().__init__()
        # class_weights is a tensor; keep it out of the hparams yaml
        self.save_hyperparameters(ignore=['class_weights'])
        self.model = KilonovaTransformer(
            d_model=d_model, num_heads=num_heads, num_layers=num_layers,
            d_feedforward=d_feedforward, dropout=dropout, num_classes=num_classes,
        )
        self.loss_function = nn.CrossEntropyLoss(weight=class_weights)
        self.train_accuracy = MulticlassAccuracy(num_classes=num_classes)
        # one accuracy metric per balanced-validation regime; the monitor is their mean
        self.val_regime_names = list(val_regime_names)
        self.validation_accuracies = nn.ModuleDict(
            {name: MulticlassAccuracy(num_classes=num_classes) for name in self.val_regime_names})
        self.validation_loss = MeanMetric()

    def forward(self, batch):
        return self.model(batch)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # the collate dict carries extra non-model keys (cid, scenario, ...); move only what the model uses
        return {key: value.to(device) for key, value in batch.items() if key in MODEL_INPUT_KEYS}

    def _step(self, batch):
        logits = self(batch)
        loss = self.loss_function(logits, batch['label'])
        return loss, logits, batch['label']

    def training_step(self, batch, batch_index):
        loss, logits, labels = self._step(batch)
        self.train_accuracy(logits, labels)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=labels.shape[0])
        self.log('train_acc', self.train_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_index, dataloader_idx=0):
        loss, logits, labels = self._step(batch)
        name = self.val_regime_names[dataloader_idx]
        self.validation_accuracies[name](logits, labels)
        self.validation_loss.update(loss, weight=labels.shape[0])  # one val_loss across all regimes
        return loss

    def on_validation_epoch_end(self):
        regime_accuracies = {}
        for name in self.val_regime_names:
            regime_accuracies[name] = self.validation_accuracies[name].compute()
            self.validation_accuracies[name].reset()
        for name, value in regime_accuracies.items():
            self.log(f'val_acc_{name}', value)

        # split the regimes by redshift availability (names are '{epochs}ep_z' / '{epochs}ep_noz').
        # with-z drives model selection (early stopping + checkpoint); no-z is the honest
        # "how it looks without redshift" number we watch but do not optimize for.
        z_values = [v for name, v in regime_accuracies.items() if not name.endswith('noz')]
        noz_values = [v for name, v in regime_accuracies.items() if name.endswith('noz')]
        balanced = torch.stack(list(regime_accuracies.values())).mean()
        self.log('val_acc_balanced', balanced)
        self.log('val_acc_z', torch.stack(z_values).mean() if z_values else balanced, prog_bar=True)
        if noz_values:
            self.log('val_acc_noz', torch.stack(noz_values).mean(), prog_bar=True)
        self.log('val_loss', self.validation_loss.compute(), prog_bar=True)
        self.validation_loss.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate,
                                      weight_decay=self.hparams.weight_decay)
        # linear warmup from min_lr up to the base lr, then cosine decay back down to min_lr.
        # transformers optimize more stably with a short warmup; cosine spans the remaining
        # epochs so it still reaches eta_min at the final epoch.
        warmup_epochs = min(self.hparams.warmup_epochs, max(self.hparams.max_epochs - 1, 0))
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(self.hparams.max_epochs - warmup_epochs, 1),
            eta_min=self.hparams.min_learning_rate)
        if warmup_epochs > 0:
            start_factor = max(self.hparams.min_learning_rate / self.hparams.learning_rate, 1e-8)
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_epochs)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        else:
            scheduler = cosine
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}


def train(data_dir, epochs, batch_size, learning_rate, weight_decay,
          checkpoint_dir, num_workers, seed, weight_mode, d_model, num_heads,
          num_layers, d_feedforward, dropout, patience, warmup_epochs):
    L.seed_everything(seed)

    data = build_dataloaders(
        deep_hdf5=os.path.join(data_dir, 'kilonova_windows_deep.hdf5'),
        wide_hdf5=os.path.join(data_dir, 'kilonova_windows_wide.hdf5'),
        deep_parquet=os.path.join(data_dir, 'early_windows_deep.parquet'),
        wide_parquet=os.path.join(data_dir, 'early_windows_wide.parquet'),
        batch_size=batch_size,
        num_workers=num_workers,
        cache_path=os.path.join(data_dir, 'openuniverse_tokens.npz'),
    )
    train_loader = data['train_loader']
    regime_loaders = data['validation_regime_loaders']
    validation_loaders = [regime['loader'] for regime in regime_loaders]
    regime_names = [regime['name'] for regime in regime_loaders]
    print(f"split sizes: {data['split_sizes']}")
    print(f"class balance: {data['class_balance']}")
    print(f"normalization: {data['normalization']}")
    print(f"validation regimes: {regime_names}")

    weights = class_weights_from_loader(train_loader, num_classes=len(GROUP_ORDER), mode=weight_mode)
    print(f'class weights ({weight_mode}) {GROUP_ORDER}: {weights.tolist()}')

    lit_model = LitKilonova(
        class_weights=weights, learning_rate=learning_rate, weight_decay=weight_decay,
        d_model=d_model, num_heads=num_heads, num_layers=num_layers,
        d_feedforward=d_feedforward, dropout=dropout, max_epochs=epochs,
        warmup_epochs=warmup_epochs, val_regime_names=regime_names,
    )
    number_of_parameters = sum(p.numel() for p in lit_model.model.parameters())
    print(f'model parameters: {number_of_parameters:,}')

    # select on val_acc_noz (no-redshift), the honest metric: KN and contaminants are nearly
    # disjoint in z (KN capped at z=0.4, contaminants peak at z~1.25), so val_acc_z=1.0 is a
    # redshift population artifact, not real skill. z is still fed to the model as an optional
    # feature in training; we just don't let it drive model selection.
    # save the top-5 so we can average their weights (model soup) instead of trusting a single
    # best epoch, which can be a lucky-validation outlier.
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='kilonova_transformer-{epoch:02d}-{val_acc_noz:.4f}',
        monitor='val_acc_noz',
        mode='max',
        save_top_k=5,
    )
    early_stopping = EarlyStopping(monitor='val_acc_noz', mode='max', patience=patience)
    learning_rate_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator='auto',
        devices='auto',
        precision='bf16-mixed',  # Ada (RTX 4060) bf16: faster, frees memory; sequences are tiny
        callbacks=[checkpoint_callback, early_stopping, learning_rate_monitor],
        log_every_n_steps=10,
    )
    trainer.fit(lit_model, train_loader, validation_loaders)

    print(f'best single val_acc_noz: {checkpoint_callback.best_model_score:.4f}  ->  '
          f'{checkpoint_callback.best_model_path}')

    # model soup: average the weights of the top-5 checkpoints into one model and evaluate it.
    # more robust than the single best epoch (which can be a validation outlier).
    soup_model, individual_scores = build_model_soup(
        checkpoint_callback.best_k_models, weights, lit_model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    soup_score, soup_per_regime = evaluate_noz(soup_model, regime_loaders, device)
    print(f'top-{len(individual_scores)} individual val_acc_noz (checkpointed): '
          f'{[round(s, 4) for s in individual_scores]}')
    print(f'souped val_acc_noz: {soup_score:.4f}  (per noz-regime: '
          f'{[round(s, 4) for s in soup_per_regime]})')
    soup_path = os.path.join(checkpoint_dir, 'kilonova_transformer-soup.ckpt')
    trainer.save_checkpoint(soup_path)  # lit_model now holds the souped weights
    print(f'souped model saved -> {soup_path}')


def build_model_soup(best_k_models, class_weights, lit_model):
    """Uniform-average the inner-transformer weights of the top-k checkpoints into lit_model.
    Returns (lit_model with souped weights, the per-checkpoint val_acc_noz scores)."""
    paths = list(best_k_models.keys())
    scores = [float(value) for value in best_k_models.values()]
    state_dicts = [
        LitKilonova.load_from_checkpoint(path, class_weights=class_weights).model.state_dict()
        for path in paths
    ]
    averaged = {
        key: sum(state_dict[key].float() for state_dict in state_dicts) / len(state_dicts)
        for key in state_dicts[0]
    }
    lit_model.model.load_state_dict(averaged)
    return lit_model, scores


def evaluate_noz(lit_model, regime_loaders, device):
    """Mean macro-accuracy over the no-redshift validation regimes (the honest metric)."""
    lit_model = lit_model.to(device).eval()
    accuracies = []
    for regime in regime_loaders:
        if not regime['name'].endswith('noz'):
            continue
        accuracy = MulticlassAccuracy(num_classes=len(GROUP_ORDER)).to(device)
        with torch.no_grad():
            for batch in regime['loader']:
                model_batch = {key: value.to(device) for key, value in batch.items()
                               if key in MODEL_INPUT_KEYS}
                accuracy(lit_model(model_batch), model_batch['label'])
        accuracies.append(accuracy.compute().item())
    return sum(accuracies) / len(accuracies), accuracies


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data/openuniverse',
                        help='directory holding the kilonova_windows_*.hdf5 and early_windows_*.parquet')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--checkpoint-dir', default='checkpoints')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--weight-mode', choices=['inverse', 'sqrt', 'none'], default='sqrt',
                        help="class-weight scheme: full inverse-freq, softened sqrt, or uniform")
    parser.add_argument('--patience', type=int, default=25,
                        help='early-stopping patience on val_acc (epochs)')
    parser.add_argument('--d-model', type=int, default=192)
    parser.add_argument('--num-heads', type=int, default=6)
    parser.add_argument('--num-layers', type=int, default=6)
    parser.add_argument('--d-feedforward', type=int, default=768,
                        help='FFN width; standard transformer ratio is 4x d_model')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='linear LR warmup length before cosine decay (epochs)')
    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_arguments()
    train(
        data_dir=arguments.data_dir,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        checkpoint_dir=arguments.checkpoint_dir,
        num_workers=arguments.num_workers,
        seed=arguments.seed,
        weight_mode=arguments.weight_mode,
        d_model=arguments.d_model,
        num_heads=arguments.num_heads,
        num_layers=arguments.num_layers,
        d_feedforward=arguments.d_feedforward,
        dropout=arguments.dropout,
        patience=arguments.patience,
        warmup_epochs=arguments.warmup_epochs,
    )
