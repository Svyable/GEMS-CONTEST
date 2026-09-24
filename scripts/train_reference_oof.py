from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import rasterio
import yaml

from gems.reference_baseline import (
    assemble_nonoverlap_patches,
    load_reference_arrays,
    make_reference_split,
    normalize_reference_features,
    patchify,
    reference_padding,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the organizer U-Net Monte Carlo OOF reference baseline"
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/reference_unet.yaml")
    parser.add_argument("--metrics-json")
    args = parser.parse_args()

    try:
        import segmentation_models_pytorch as smp
        import torch
        from segmentation_models_pytorch.losses import TverskyLoss
        from torch import optim
        from torch.nn import functional
        from torch.utils.data import DataLoader, TensorDataset
        from torchvision import tv_tensors
        from torchvision.transforms import v2
        from torchvision.transforms.functional import InterpolationMode
    except ImportError as exc:
        raise SystemExit(
            "ML dependencies are missing. Install with "
            "uv sync --extra ml --extra cpu --extra dev "
            "(or replace cpu with cu126)."
        ) from exc

    config = yaml.safe_load(Path(args.config).read_text())
    features, labels, _ = load_reference_arrays(args.features, args.labels)
    features = normalize_reference_features(features)

    patch_size = int(config["patches"]["patch_size"])
    train_step = int(config["patches"]["train_step"])
    test_proportion = float(config["patches"]["test_proportion"])
    seeds = [int(value) for value in config["seed_sequence"]]
    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])
    learning_rate = float(config["training"]["learning_rate"])
    alpha = float(config["training"]["alpha"])
    beta = float(config["training"]["beta"])

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device={device}")

    pad_y, pad_x = reference_padding(labels.shape, patch_size)
    padded_labels = np.pad(
        labels,
        ((0, pad_y), (0, pad_x)),
        mode="constant",
        constant_values=0,
    )
    label_tiles = patchify(
        padded_labels,
        (patch_size, patch_size),
        patch_size,
    )
    prediction_tiles = np.zeros(label_tiles.shape, dtype=np.float32)

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
            v2.RandomRotation(
                float(config["augmentation"]["random_rotation_degrees"])
            ),
        ]
    )

    run_metrics: list[dict] = []
    for mc_index, seed in enumerate(seeds):
        print(f"split={mc_index + 1}/{len(seeds)} seed={seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        split = make_reference_split(
            features,
            labels,
            patch_size=patch_size,
            test_proportion=test_proportion,
            seed=seed,
            train_step=train_step,
        )
        if split.x_train.shape[0] == 0 or split.x_test.shape[0] == 0:
            raise SystemExit("reference split produced an empty train or test set")
        print(
            f"train_patches={split.x_train.shape[0]} "
            f"test_patches={split.x_test.shape[0]}"
        )

        x_train = torch.from_numpy(split.x_train).float()
        y_train = torch.from_numpy(split.y_train).float()
        x_test = torch.from_numpy(split.x_test).float()
        y_test = torch.from_numpy(split.y_test).float()

        train_loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            TensorDataset(x_test, y_test),
            batch_size=batch_size,
            shuffle=False,
        )

        model = smp.Unet(
            encoder_name=config["model"]["encoder"],
            encoder_weights=config["model"]["encoder_weights"],
            in_channels=x_train.shape[1],
            classes=int(config["model"]["classes"]),
        ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = TverskyLoss(alpha=alpha, beta=beta, mode="binary")

        best_loss = float("inf")
        best_state = None
        epochs_log: list[dict] = []

        for epoch in range(epochs):
            model.train()
            train_loss_total = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_x, batch_y = transform(batch_x, tv_tensors.Mask(batch_y))
                outputs = model(batch_x)[:, 0]
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss_total += float(loss.item())

            model.eval()
            test_loss_total = 0.0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    outputs = model(batch_x)[:, 0]
                    test_loss_total += float(criterion(outputs, batch_y).item())

            train_loss = train_loss_total / len(train_loader)
            test_loss = test_loss_total / len(test_loader)
            epochs_log.append(
                {"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss}
            )
            print(
                f"epoch={epoch} train_loss={train_loss:.6f} "
                f"test_loss={test_loss:.6f}"
            )
            if test_loss < best_loss:
                best_loss = test_loss
                best_state = copy.deepcopy(model.state_dict())

        if best_state is None:
            raise RuntimeError("no model state was selected")
        model.load_state_dict(best_state)
        model.eval()

        batches = []
        with torch.no_grad():
            for batch_x, _ in test_loader:
                logits = model(batch_x.to(device))[:, 0]
                batches.append(functional.sigmoid(logits).cpu())
        test_predictions = torch.cat(batches, dim=0).numpy()

        for index, (row, col) in enumerate(split.test_indices):
            prediction_tiles[row, col] += test_predictions[index] / len(seeds)

        run_metrics.append(
            {
                "seed": seed,
                "train_patches": int(split.x_train.shape[0]),
                "test_patches": int(split.x_test.shape[0]),
                "best_test_loss": best_loss,
                "epochs": epochs_log,
            }
        )

    combined = assemble_nonoverlap_patches(prediction_tiles)
    prediction = combined[: labels.shape[0], : labels.shape[1]].astype(np.float32)

    with rasterio.open(args.labels) as label_src:
        profile = label_src.profile.copy()
        valid_mask = label_src.dataset_mask() > 0
    prediction[~valid_mask] = np.nan
    profile.update(count=1, dtype="float32", nodata=np.nan, compress="deflate")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(prediction, 1)
    print(f"wrote {output}")

    if args.metrics_json:
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps({"runs": run_metrics}, indent=2) + "\n")
        print(f"wrote {metrics_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
