from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import yaml

from gems.prediction import write_prediction_like_template
from gems.reference_baseline import load_reference_arrays, normalize_reference_features
from gems.samples import training_origins
from gems.submission import validate_submission
from gems.tiling import blend_predictions, extract_patch, generate_windows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train one U-Net on all known faults and predict the full study area"
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/reference_unet.yaml")
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--metrics-json")
    args = parser.parse_args()

    try:
        import torch
        from segmentation_models_pytorch import Unet
        from segmentation_models_pytorch.losses import TverskyLoss
        from torch import optim
        from torch.nn import functional
        from torch.utils.data import DataLoader, Dataset
        from torchvision import tv_tensors
        from torchvision.transforms import v2
        from torchvision.transforms.functional import InterpolationMode
    except ImportError as exc:
        raise SystemExit(
            "ML dependencies are missing. Install with "
            "uv sync --extra ml --extra cpu --extra dev."
        ) from exc

    config = yaml.safe_load(Path(args.config).read_text())
    patch_size = int(config["patches"]["patch_size"])
    step = int(config["patches"]["train_step"])
    epochs = int(config["training"]["epochs"])
    batch_size = int(config["training"]["batch_size"])
    learning_rate = float(config["training"]["learning_rate"])
    alpha = float(config["training"]["alpha"])
    beta = float(config["training"]["beta"])

    features, labels, _ = load_reference_arrays(args.features, args.labels)
    features = np.nan_to_num(normalize_reference_features(features)).astype(np.float32)
    with rasterio.open(args.features) as src:
        valid = src.dataset_mask() > 0

    origins = training_origins(
        labels,
        valid,
        patch_size=patch_size,
        step=step,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
    )
    if not origins:
        raise SystemExit("no training windows were selected")
    n_positive = sum(
        1
        for row, col in origins
        if np.any(labels[row : row + patch_size, col : col + patch_size])
    )
    print(
        f"windows={len(origins)} positive={n_positive} "
        f"negative={len(origins) - n_positive}"
    )

    class WindowDataset(Dataset):
        def __len__(self) -> int:
            return len(origins)

        def __getitem__(self, index: int):
            row, col = origins[index]
            patch = features[row : row + patch_size, col : col + patch_size]
            target = labels[row : row + patch_size, col : col + patch_size]
            return (
                torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(target)),
            )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device={device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        WindowDataset(),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    model = Unet(
        encoder_name=config["model"]["encoder"],
        encoder_weights=config["model"]["encoder_weights"],
        in_channels=features.shape[-1],
        classes=int(config["model"]["classes"]),
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = TverskyLoss(alpha=alpha, beta=beta, mode="binary")
    transform = v2.Compose(
        [
            v2.RandomResizedCrop(
                patch_size,
                scale=tuple(config["augmentation"]["random_resized_crop"]["scale"]),
                ratio=tuple(config["augmentation"]["random_resized_crop"]["ratio"]),
                interpolation=InterpolationMode.BILINEAR,
            ),
            v2.RandomHorizontalFlip(
                p=float(config["augmentation"]["horizontal_flip_probability"])
            ),
            v2.RandomVerticalFlip(
                p=float(config["augmentation"]["vertical_flip_probability"])
            ),
            v2.RandomRotation(float(config["augmentation"]["random_rotation_degrees"])),
        ]
    )

    history = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_x, batch_y = transform(batch_x, tv_tensors.Mask(batch_y))
            loss = criterion(model(batch_x)[:, 0], batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += float(loss.item())
        mean_loss = total / len(loader)
        history.append({"epoch": epoch, "train_loss": mean_loss})
        print(f"epoch={epoch} train_loss={mean_loss:.6f}")

    windows = generate_windows(labels.shape, patch_size=patch_size, overlap=args.overlap)
    model.eval()
    predictions: list[np.ndarray] = []
    batch: list[np.ndarray] = []
    with torch.no_grad():
        for window in windows:
            patch = extract_patch(features, window, patch_size=patch_size, fill_value=0.0)
            batch.append(np.ascontiguousarray(patch.transpose(2, 0, 1)))
            if len(batch) == batch_size:
                tensor = torch.from_numpy(np.stack(batch)).to(device)
                probabilities = functional.sigmoid(model(tensor)[:, 0]).cpu().numpy()
                predictions.extend(probabilities)
                batch = []
        if batch:
            tensor = torch.from_numpy(np.stack(batch)).to(device)
            probabilities = functional.sigmoid(model(tensor)[:, 0]).cpu().numpy()
            predictions.extend(probabilities)

    blended = blend_predictions(
        labels.shape,
        windows,
        predictions,
        patch_size=patch_size,
        valid_mask=valid,
    )
    blended = np.nan_to_num(blended, nan=0.0)
    blended = np.clip(blended, 0.0, 1.0).astype(np.float32)
    path = write_prediction_like_template(
        blended,
        template_path=args.template,
        output_path=args.output,
    )
    report = validate_submission(path, args.template)
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for error in report.errors:
        print(f"ERROR: {error}")
    if not report.ok:
        return 1
    print(f"OK: wrote validated full-map prediction to {path}")

    if args.metrics_json:
        payload = {
            "windows": len(origins),
            "positive_windows": n_positive,
            "negative_windows": len(origins) - n_positive,
            "epochs": history,
            "overlap": args.overlap,
            "seed": args.seed,
            "negative_ratio": args.negative_ratio,
        }
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
