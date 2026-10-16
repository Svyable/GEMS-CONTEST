from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import yaml

from gems.data import raster_alignment_errors, sha256_file
from gems.distance_tversky import build_training_loss
from gems.lineament import (
    grouped_lineament_features,
    lineament_kwargs,
    raster_category_groups,
    required_lineament_buffer,
)
from gems.metric import distance_weighted_tversky
from gems.prediction import write_prediction_like_template
from gems.preprocessing import normalize_training_features
from gems.reference_baseline import load_reference_arrays
from gems.robustness import NEIGHBOR_SHIFTS, shift_feature_channels, windows_intersecting_mask
from gems.samples import training_origins
from gems.submission import validate_submission
from gems.tiling import blend_predictions, extract_patch, generate_windows
from gems.training_split import candidate_training_split, masked_training_arrays


def masked_binary_loss(criterion, logits, targets, supervision):
    """Apply the same binary criterion only to supervised pixels across the batch."""
    if not supervision.any():
        return None
    return criterion(logits[supervision].reshape(1, 1, -1),
                     targets[supervision].reshape(1, 1, -1))


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
    parser.add_argument(
        "--fold-map", help="Spatial, complete-fault or endpoint fold GeoTIFF."
    )
    parser.add_argument("--cv-scheme", choices=("spatial", "fault", "trace"), default="spatial")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--buffer-pixels", type=int, default=16)
    parser.add_argument("--known-fault-exclusion-pixels", type=int, default=0)
    parser.add_argument("--min-training-fraction", type=float, default=0.5,
                        help="Minimum supervised window fraction for fault/trace CV only")
    parser.add_argument("--metrics-json")
    parser.add_argument(
        "--registration-sensitivity-json",
        help="Write one-pixel feature-family registration stress-test results (spatial CV only).",
    )
    args = parser.parse_args()
    if not args.fold_map and (args.cv_scheme != "spatial" or args.known_fault_exclusion_pixels):
        raise SystemExit("fault/trace validation requires --fold-map")

    config = yaml.safe_load(Path(args.config).read_text())
    patch_size = int(config["patches"]["patch_size"])
    step = int(config["patches"]["train_step"])
    epochs = int(config["training"]["epochs"])
    batch_size = int(config["training"]["batch_size"])
    learning_rate = float(config["training"]["learning_rate"])
    alpha = float(config["training"]["alpha"])
    beta = float(config["training"]["beta"])

    for name, path in [
        ("labels", args.labels),
        ("template", args.template),
        ("fold map", args.fold_map),
    ]:
        if path:
            errors = raster_alignment_errors(args.features, path)
            if errors:
                raise SystemExit(f"{name}: " + "; ".join(errors))

    features, labels, _ = load_reference_arrays(args.features, args.labels)
    with rasterio.open(args.features) as src:
        valid = src.dataset_mask() > 0

    split = None
    allowed = None
    if args.fold_map:
        with rasterio.open(args.fold_map) as src:
            if src.count != 1 or not np.issubdtype(np.dtype(src.dtypes[0]), np.integer):
                raise SystemExit("fold map must be a single-band integer raster")
            fold_values = src.read(1)
        try:
            split = candidate_training_split(
                labels > 0, fold_values, scheme=args.cv_scheme, fold=args.fold,
                valid_mask=valid, buffer_pixels=args.buffer_pixels,
                known_fault_exclusion_pixels=args.known_fault_exclusion_pixels,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        allowed = split.train_mask
        print(f"fold={args.fold} split={split.summary()}")

    features, normalization = normalize_training_features(
        features, valid if allowed is None else allowed
    )

    lineament_metadata = None
    lineament_config = config.get("derived_features", {}).get("lineament", {})
    if lineament_config.get("enabled", False):
        categories = tuple(lineament_config.get("categories", ()))
        try:
            options = lineament_kwargs(lineament_config)
            if split is not None:
                required_buffer = required_lineament_buffer(options)
                if args.buffer_pixels < required_buffer:
                    raise ValueError(
                        "lineament filtering requires --buffer-pixels >= "
                        f"{required_buffer} for kind={options['kind']}"
                    )
            groups = raster_category_groups(args.features, categories)
            # Leakage-safe protocol for fault/trace CV: two-pass filtering to compute
            # derived features for both training and held-out regions without leakage.
            # Pass A computes training-region features using train_mask. Pass B computes
            # held-out features using evaluation_mask. Both passes use fold-pure
            # normalization (train_mask only). Buffer pixels remain zero.
            if split is not None:
                # Pass A: training pixels with train_mask, collect normalization parameters
                derived_train, lineament_metadata = grouped_lineament_features(
                    features,
                    groups,
                    valid_mask=split.train_mask,
                    normalization_mask=split.train_mask,
                    **options,
                )
                # Pass B: held-out pixels with evaluation_mask, reuse Pass A normalization
                norm_params = lineament_metadata.get("norm_params")
                derived_evaluation, _ = grouped_lineament_features(
                    features,
                    groups,
                    valid_mask=split.evaluation_mask,
                    normalization_mask=split.train_mask,
                    norm_params=norm_params,
                    **options,
                )
                # Combine by region: training where train_mask, evaluation where evaluation_mask
                derived = np.where(
                    split.train_mask[..., np.newaxis],
                    derived_train,
                    np.where(
                        split.evaluation_mask[..., np.newaxis],
                        derived_evaluation,
                        0.0,
                    ),
                )
                # Runtime check: ensure derived channels have similar distributions
                # across training and held-out regions (catch scale mismatch bugs)
                for channel_idx in range(derived.shape[-1]):
                    train_values = derived[split.train_mask, channel_idx]
                    eval_values = derived[split.evaluation_mask, channel_idx]
                    channel_name = lineament_metadata["output_channels"][channel_idx]
                    
                    if eval_values.size == 0:
                        continue
                    
                    # Check for constant zero (original bug)
                    if eval_values.min() == eval_values.max() == 0.0:
                        raise ValueError(
                            f"derived channel {channel_name!r} is constant zero over "
                            f"held-out region (fold={args.fold}); normalization failed"
                        )
                    
                    # Check for scale mismatch (>5x difference in median or p99)
                    if train_values.size > 0:
                        train_median = float(np.median(train_values))
                        eval_median = float(np.median(eval_values))
                        train_p99 = float(np.percentile(train_values, 99.0))
                        eval_p99 = float(np.percentile(eval_values, 99.0))
                        
                        # Compare ratios (handle near-zero denominators)
                        median_ratio = (
                            eval_median / train_median if train_median > 1e-6
                            else float('inf') if eval_median > 1e-6
                            else 1.0
                        )
                        p99_ratio = (
                            eval_p99 / train_p99 if train_p99 > 1e-6
                            else float('inf') if eval_p99 > 1e-6
                            else 1.0
                        )
                        
                        if median_ratio > 5.0 or median_ratio < 0.2:
                            raise ValueError(
                                f"derived channel {channel_name!r} has {median_ratio:.1f}x "
                                f"median mismatch (train={train_median:.4f}, "
                                f"eval={eval_median:.4f}, fold={args.fold}); "
                                f"normalization scale reuse failed"
                            )
                        
                        if p99_ratio > 5.0 or p99_ratio < 0.2:
                            raise ValueError(
                                f"derived channel {channel_name!r} has {p99_ratio:.1f}x "
                                f"p99 mismatch (train={train_p99:.4f}, "
                                f"eval={eval_p99:.4f}, fold={args.fold}); "
                                f"normalization scale reuse failed"
                            )
            else:
                # Spatial CV: single pass with full valid mask (no leakage concern)
                derived, lineament_metadata = grouped_lineament_features(
                    features,
                    groups,
                    valid_mask=valid,
                    normalization_mask=allowed if allowed is not None else valid,
                    **options,
                )
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"lineament features: {exc}") from exc
        base_channels = features.shape[-1]
        features = np.concatenate([features, derived], axis=-1)
        lineament_metadata["base_channels"] = int(base_channels)
        lineament_metadata["derived_channels"] = int(derived.shape[-1])
        lineament_metadata["total_channels"] = int(features.shape[-1])
        print(
            "lineament_features="
            f"{lineament_metadata['kind']} "
            f"derived_channels={derived.shape[-1]} total_channels={features.shape[-1]}"
        )

    masked_supervision = split is not None and args.cv_scheme in ("fault", "trace")
    train_features, train_labels, supervision = features, labels, valid
    if masked_supervision:
        train_features, train_labels, supervision = masked_training_arrays(features, split)

    origins = training_origins(
        train_labels,
        valid,
        patch_size=patch_size,
        step=step,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
        allowed_mask=allowed,
        min_allowed_fraction=args.min_training_fraction if masked_supervision else 1.0,
    )
    if not origins:
        raise SystemExit("no training windows were selected")
    n_positive = sum(
        1 for row, col in origins
        if np.any(train_labels[row : row + patch_size, col : col + patch_size])
    )
    print(f"windows={len(origins)} positive={n_positive} negative={len(origins) - n_positive}")

    try:
        import torch
        from segmentation_models_pytorch import Unet
        from torch import optim
        from torch.nn import functional
        from torch.utils.data import DataLoader, Dataset
        from torchvision import tv_tensors
        from torchvision.transforms import v2
        from torchvision.transforms.functional import InterpolationMode
    except ImportError as exc:
        raise SystemExit(
            "ML dependencies are missing. Install with uv sync --extra ml --extra cpu --extra dev."
        ) from exc

    class WindowDataset(Dataset):
        def __len__(self) -> int:
            return len(origins)

        def __getitem__(self, index: int):
            row, col = origins[index]
            patch = train_features[row : row + patch_size, col : col + patch_size]
            target = train_labels[row : row + patch_size, col : col + patch_size]
            result = (
                torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(target)),
            )
            if masked_supervision:
                mask = supervision[row : row + patch_size, col : col + patch_size]
                return (*result, torch.from_numpy(np.ascontiguousarray(mask)))
            return result

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device={device} encoder={config['model']['encoder']}")

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
    criterion, spatial_loss = build_training_loss(config)
    loss_descriptor = {
        "name": str(config["training"].get("loss", "tversky")),
        "alpha": alpha,
        "beta": beta,
        "distance_weight": float(config["training"].get("loss_distance_weight", 0.5)),
        "radius_pixels": float(config["training"].get("loss_radius_pixels", 3.0)),
        "halo_pixels": int(config["training"].get("loss_halo_pixels", 0)),
        "background_soften_factor": float(config["training"].get("loss_background_soften_factor", 0.5)),
        "ambiguity_radius_pixels": float(config["training"].get("loss_ambiguity_radius_pixels", 3.0)),
        "skeleton_weight": float(config["training"].get("loss_skeleton_weight", 0.25)),
        "skeleton_iterations": int(config["training"].get("loss_skeleton_iterations", 5)),
    }
    transform = v2.Compose(
        [
            v2.RandomResizedCrop(
                patch_size,
                scale=tuple(config["augmentation"]["random_resized_crop"]["scale"]),
                ratio=tuple(config["augmentation"]["random_resized_crop"]["ratio"]),
                interpolation=InterpolationMode.BILINEAR,
            ),
            v2.RandomHorizontalFlip(p=float(config["augmentation"]["horizontal_flip_probability"])),
            v2.RandomVerticalFlip(p=float(config["augmentation"]["vertical_flip_probability"])),
            v2.RandomRotation(float(config["augmentation"]["random_rotation_degrees"])),
        ]
    )

    history = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        steps, supervised_pixels = 0, 0
        for batch in loader:
            batch_x, batch_y = batch[:2]
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            if masked_supervision:
                mask = batch[2].to(device)
                targets = torch.stack([batch_y, mask.to(batch_y.dtype)], dim=1)
                batch_x, targets = transform(batch_x, tv_tensors.Mask(targets))
                batch_y, mask = targets[:, 0], targets[:, 1].bool()
                if not mask.any():
                    continue
                batch_x = batch_x.masked_fill(~mask[:, None], 0)
                if spatial_loss:
                    # Distance-aware losses keep the 2D batch structure: the
                    # supervision mask zeroes contributions instead of
                    # flattening, so the distance kernel stays meaningful.
                    loss = criterion(model(batch_x)[:, 0], batch_y, mask)
                else:
                    loss = masked_binary_loss(criterion, model(batch_x)[:, 0], batch_y, mask)
                supervised_pixels += int(mask.sum().item())
            else:
                batch_x, batch_y = transform(batch_x, tv_tensors.Mask(batch_y))
                if spatial_loss:
                    loss = criterion(model(batch_x)[:, 0], batch_y)
                else:
                    loss = criterion(model(batch_x)[:, 0], batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += float(loss.item())
            steps += 1
        if not steps:
            raise SystemExit("augmentation left no supervised training batches")
        mean_loss = total / steps
        history.append({"epoch": epoch, "train_loss": mean_loss, "optimizer_steps": steps,
                        "supervised_pixels": supervised_pixels if masked_supervision else None})
        print(f"epoch={epoch} train_loss={mean_loss:.6f}")

    windows = generate_windows(labels.shape, patch_size=patch_size, overlap=args.overlap)
    model.eval()

    def infer_windows(feature_stack, selected_windows):
        predictions: list[np.ndarray] = []
        batch: list[np.ndarray] = []
        with torch.no_grad():
            for window in selected_windows:
                patch = extract_patch(
                    feature_stack, window, patch_size=patch_size, fill_value=0.0
                )
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
        return predictions

    predictions = infer_windows(features, windows)
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

    holdout_score = None
    if split is not None:
        holdout_score = float(distance_weighted_tversky(
            blended, split.evaluation_truth, valid_mask=split.evaluation_mask
        ))
        print(f"holdout_distance_weighted_tversky={holdout_score:.8f}")

    if args.registration_sensitivity_json:
        if split is None or args.cv_scheme != "spatial":
            raise SystemExit("registration sensitivity requires --fold-map with --cv-scheme spatial")
        if lineament_metadata is not None:
            raise SystemExit(
                "registration sensitivity currently requires a base-feature config without "
                "derived lineament channels"
            )
        sensitivity_config = config.get("robustness", {}).get(
            "registration_sensitivity", {}
        )
        sensitivity_categories = tuple(
            sensitivity_config.get(
                "categories",
                ("magnetic_data", "gravity_data", "geodetic_strain", "topographic"),
            )
        )
        try:
            sensitivity_groups = raster_category_groups(
                args.features, sensitivity_categories
            )
        except ValueError as exc:
            raise SystemExit(f"registration sensitivity: {exc}") from exc

        evaluation_windows = windows_intersecting_mask(windows, split.evaluation_mask)
        baseline_eval = blend_predictions(
            labels.shape,
            evaluation_windows,
            infer_windows(features, evaluation_windows),
            patch_size=patch_size,
            valid_mask=split.evaluation_mask,
        )
        baseline_eval = np.nan_to_num(baseline_eval, nan=0.0)
        baseline_eval_score = float(
            distance_weighted_tversky(
                baseline_eval,
                split.evaluation_truth,
                valid_mask=split.evaluation_mask,
            )
        )
        if holdout_score is None or not np.isclose(
            baseline_eval_score, holdout_score, rtol=0, atol=1e-7
        ):
            raise RuntimeError(
                "evaluation-window filtering changed the baseline holdout score"
            )

        sensitivity_results = {}
        for category, channels in sensitivity_groups.items():
            shifts = []
            for row_offset, col_offset in NEIGHBOR_SHIFTS:
                perturbed = shift_feature_channels(
                    features,
                    channels,
                    row_offset=row_offset,
                    col_offset=col_offset,
                    valid_mask=valid,
                    fill_value=0.0,
                )
                perturbed_eval = blend_predictions(
                    labels.shape,
                    evaluation_windows,
                    infer_windows(perturbed, evaluation_windows),
                    patch_size=patch_size,
                    valid_mask=split.evaluation_mask,
                )
                perturbed_eval = np.nan_to_num(perturbed_eval, nan=0.0)
                score = float(
                    distance_weighted_tversky(
                        perturbed_eval,
                        split.evaluation_truth,
                        valid_mask=split.evaluation_mask,
                    )
                )
                shifts.append(
                    {
                        "row_offset": row_offset,
                        "col_offset": col_offset,
                        "score": score,
                        "delta_from_baseline": score - baseline_eval_score,
                    }
                )
                print(
                    "registration_sensitivity "
                    f"category={category} shift=({row_offset},{col_offset}) "
                    f"score={score:.8f} delta={score - baseline_eval_score:+.8f}"
                )
            worst = min(shifts, key=lambda item: item["score"])
            sensitivity_results[category] = {
                "channels_zero_based": list(channels),
                "shifts": shifts,
                "worst_score": worst["score"],
                "worst_delta_from_baseline": worst["delta_from_baseline"],
            }

        with rasterio.open(args.features) as src:
            resolution = [float(src.res[0]), float(src.res[1])]
        sensitivity_payload = {
            "schema_version": 1,
            "method": "one_pixel_feature_family_translation_v1",
            "fold": args.fold,
            "baseline_score": baseline_eval_score,
            "evaluation_windows": len(evaluation_windows),
            "pixel_resolution": resolution,
            "fill_value": 0.0,
            "categories": sensitivity_results,
            "config_sha256": sha256_file(args.config),
            "input_sha256": {
                "features": sha256_file(args.features),
                "labels": sha256_file(args.labels),
                "fold_map": sha256_file(args.fold_map),
            },
        }
        sensitivity_path = Path(args.registration_sensitivity_json)
        sensitivity_path.parent.mkdir(parents=True, exist_ok=True)
        sensitivity_path.write_text(json.dumps(sensitivity_payload, indent=2) + "\n")
        print(f"wrote {sensitivity_path}")

    if args.metrics_json:
        payload = {
            "normalization": normalization,
            "derived_features": {"lineament": lineament_metadata}
            if lineament_metadata is not None
            else None,
            "feature_channels": {
                "base": len(normalization["minimum"]),
                "derived": 0 if lineament_metadata is None else lineament_metadata["derived_channels"],
                "total": features.shape[-1],
            },
            "config_sha256": sha256_file(args.config),
            "input_sha256": {
                name: sha256_file(path)
                for name, path in [
                    ("features", args.features),
                    ("labels", args.labels),
                    ("template", args.template),
                    ("fold_map", args.fold_map),
                ]
                if path
            },
            "buffer_pixels": args.buffer_pixels if args.fold_map else None,
            "validation": split.summary() if split else None,
            "known_fault_exclusion_pixels": args.known_fault_exclusion_pixels,
            "training_input_mask": "zero_excluded_pixels" if masked_supervision else None,
            "training_loss_mask": "supervised_pixels_only" if masked_supervision else None,
            "training_loss": loss_descriptor,
            "min_training_fraction": args.min_training_fraction if masked_supervision else 1.0,
            "encoder": config["model"]["encoder"],
            "windows": len(origins),
            "positive_windows": n_positive,
            "negative_windows": len(origins) - n_positive,
            "epochs": history,
            "overlap": args.overlap,
            "seed": args.seed,
            "train_step": step,
            "negative_ratio": args.negative_ratio,
            "fold": args.fold if args.fold_map else None,
            "holdout_distance_weighted_tversky": holdout_score,
        }
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
