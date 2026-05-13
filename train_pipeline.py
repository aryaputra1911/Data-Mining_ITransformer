"""
GEMASTIK 2026 -- iTransformer Training Pipeline v4

Deep Optimization:
  1. 48-dim input: 33 price + 15 temporal context
  2. Dice+Focal hybrid loss with sqrt(pos_weight)
  3. Learnable temperature attention + RevIN + TemporalEmbedding
  4. SWA for last 30%, batch_size=64, AUPRC metric
  5. Automated threshold calibration on val set

Usage:
    python train_pipeline.py                       # Train all 4 models
    python train_pipeline.py --model iTransformer  # Train single model
    python train_pipeline.py --ablation            # Run ablation study
    python train_pipeline.py --analysis-only       # Attention analysis only
"""
import argparse
import time
import torch
import numpy as np
import pandas as pd

from config import (
    DEVICE, CKPT_DIR, RESULT_DIR, N_VARIATES, N_TEMPORAL,
    CORE_VARIATES, CORE_VARIATE_COLS, LABEL_COLS,
)
from dataset import load_datasets, make_loaders
from models import build_all_models, build_ablation_models, iTransformerModel
from losses import build_criterion
from trainer import (
    train_model, evaluate, evaluate_with_threshold_search,
    save_per_variate_mae, evaluate_price_space, _split_input,
)
from analysis import (
    extract_attention_matrix, plot_attention_heatmap,
    build_propagation_network, plot_propagation_graph,
    save_centrality_report,
)


def build_results_table(models_results: dict) -> pd.DataFrame:
    """Build comparison table with price-space metrics + AUPRC."""
    rows = []
    for name, metrics in models_results.items():
        rows.append({
            "Model": name,
            "MAE (lr)": f"{metrics['mae']:.4f}",
            "RMSE (lr)": f"{metrics['rmse']:.4f}",
            "Price MAE (Rp)": f"{metrics.get('price_mae', 0):,.0f}",
            "Price sMAPE": f"{metrics.get('price_smape', 0):.2f}%",
            "F1 (macro)": f"{metrics['f1']:.4f}",
            "Precision": f"{metrics['precision']:.4f}",
            "Recall": f"{metrics['recall']:.4f}",
            "AUPRC": f"{metrics.get('auprc', 0):.4f}",
            "AUC-ROC": f"{metrics['auc']:.4f}",
            "Threshold": f"{metrics.get('threshold', 0.5):.2f}",
        })
    df = pd.DataFrame(rows)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULT_DIR / "comparison_table.csv", index=False)
    print("\n" + "=" * 120)
    print("  MODEL COMPARISON TABLE")
    print("=" * 120)
    print(df.to_string(index=False))
    print("=" * 120)
    return df


def run_attention_analysis(test_loader):
    """Load best iTransformer checkpoint and run full attention analysis."""
    print("\n" + "=" * 65)
    print("  ATTENTION ANALYSIS -- iTransformer")
    print("=" * 65)

    ckpt_path = CKPT_DIR / "iTransformer_best.pt"
    swa_path  = CKPT_DIR / "iTransformer_swa.pt"

    # Prefer SWA checkpoint
    if swa_path.exists():
        load_path = swa_path
    elif ckpt_path.exists():
        load_path = ckpt_path
    else:
        print(f"  [ERROR] No checkpoint found. Train iTransformer first!")
        return

    ckpt = torch.load(load_path, map_location=DEVICE, weights_only=False)
    model = iTransformerModel().to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    print(f"  Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val_loss={ckpt['val_loss']:.4f}, swa={ckpt.get('swa', False)})")

    if ckpt['epoch'] <= 3:
        print(f"  [WARN] Best checkpoint is from epoch {ckpt['epoch']}!")
        print("  Model may not have converged.")

    # Print learned temperatures
    temps = model.get_learned_temperature()
    for i, t in temps:
        print(f"    Layer {i} learned temperature: {t:.4f}")

    # Extract attention matrix
    print("\n  [1/4] Extracting attention matrix...")
    attn_matrix = extract_attention_matrix(model, test_loader, DEVICE)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(RESULT_DIR / "attention_matrix_33x33.npy", attn_matrix)
    print(f"    Shape: {attn_matrix.shape}")
    print(f"    Range: [{attn_matrix.min():.4f}, {attn_matrix.max():.4f}]")

    attn_range = attn_matrix.max() - attn_matrix.min()
    if attn_range < 0.02:
        print(f"    [WARN] Attention range ({attn_range:.4f}) is narrow -- near-uniform.")
    else:
        print(f"    [OK] Attention range ({attn_range:.4f}) shows differentiation.")

    print("\n  [2/4] Plotting attention heatmap...")
    plot_attention_heatmap(attn_matrix)

    print("\n  [3/4] Building propagation network...")
    G, out_cent, in_cent = build_propagation_network(attn_matrix)

    print("\n  [4/4] Plotting propagation graph...")
    plot_propagation_graph(G)

    save_centrality_report(out_cent, in_cent)

    print("\n  [OK] Attention analysis complete!")


def run_ablation_study(train_loader, val_loader, test_loader, criterion):
    """Run iTransformer ablation study."""
    print("\n" + "=" * 65)
    print("  ABLATION STUDY -- iTransformer")
    print("=" * 65)

    ablation_models = build_ablation_models()
    ablation_results = {}

    for name, model in ablation_models.items():
        print(f"\n  --- Ablation: {name} ---")
        trained, history = train_model(model, name, train_loader, val_loader, criterion)

        ckpt_path = CKPT_DIR / f"{name}_best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            trained.load_state_dict(ckpt['model_state'])

        _, test_metrics, opt_t = evaluate_with_threshold_search(
            trained, val_loader, test_loader, criterion, DEVICE
        )
        ablation_results[name] = test_metrics
        print(f"  {name} TEST: MAE={test_metrics['mae']:.4f}, "
              f"F1={test_metrics['f1']:.4f}, AUPRC={test_metrics['auprc']:.4f}")

    ablation_df = build_results_table(ablation_results)
    ablation_df.to_csv(RESULT_DIR / "ablation_table.csv", index=False)
    print("\n  [OK] Ablation study complete!")
    return ablation_results


def main():
    parser = argparse.ArgumentParser(description="GEMASTIK iTransformer Training Pipeline v4")
    parser.add_argument('--model', type=str, default=None,
                        help='Train single model: LSTM, PatchTST, Crossformer, iTransformer')
    parser.add_argument('--ablation', action='store_true', help='Run ablation study')
    parser.add_argument('--analysis-only', action='store_true', help='Attention analysis only')
    parser.add_argument('--skip-analysis', action='store_true', help='Skip attention analysis')
    args = parser.parse_args()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n  Device    : {DEVICE}")
    print(f"  Checkpoints: {CKPT_DIR}")
    print(f"  Results    : {RESULT_DIR}")

    train_ds, val_ds, test_ds = load_datasets()
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds)

    if args.analysis_only:
        run_attention_analysis(test_loader)
        return

    criterion = build_criterion(DEVICE)

    if args.ablation:
        run_ablation_study(train_loader, val_loader, test_loader, criterion)
        return

    all_models = build_all_models()
    if args.model:
        if args.model not in all_models:
            print(f"  [ERROR] Unknown model: {args.model}")
            print(f"  Available: {list(all_models.keys())}")
            return
        models_to_train = {args.model: all_models[args.model]}
    else:
        models_to_train = all_models

    results = {}
    total_start = time.time()

    for name, model in models_to_train.items():
        t0 = time.time()
        trained_model, history = train_model(
            model, name, train_loader, val_loader, criterion,
        )

        # Prefer SWA checkpoint
        swa_path = CKPT_DIR / f"{name}_swa.pt"
        ckpt_path = CKPT_DIR / f"{name}_best.pt"
        if swa_path.exists():
            ckpt = torch.load(swa_path, map_location=DEVICE, weights_only=False)
            trained_model.load_state_dict(ckpt['model_state'])
            print(f"\n  Loaded SWA checkpoint: epoch {ckpt['epoch']}")
        elif ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            trained_model.load_state_dict(ckpt['model_state'])
            print(f"\n  Loaded best checkpoint: epoch {ckpt['epoch']}")

        val_metrics, test_metrics, opt_t = evaluate_with_threshold_search(
            trained_model, val_loader, test_loader, criterion, DEVICE
        )

        # Fix 1: Price-space evaluation
        price_metrics = evaluate_price_space(trained_model, test_ds, DEVICE, name)
        test_metrics.update(price_metrics)
        results[name] = test_metrics

        if 'mae_per_var' in test_metrics:
            save_per_variate_mae(test_metrics['mae_per_var'], name)

        elapsed = time.time() - t0
        print(f"\n  {name} TEST RESULTS ({elapsed:.0f}s):")
        print(f"    MAE (log-return): {test_metrics['mae']:.4f}")
        print(f"    MAE (price):  Rp {test_metrics.get('price_mae', 0):,.0f}")
        print(f"    sMAPE (price): {test_metrics.get('price_smape', 0):.2f}%")
        print(f"    F1:    {test_metrics['f1']:.4f} (threshold={opt_t:.2f})")
        print(f"    Prec:  {test_metrics['precision']:.4f}")
        print(f"    Rec:   {test_metrics['recall']:.4f}")
        print(f"    AUPRC: {test_metrics['auprc']:.4f}")
        print(f"    AUC:   {test_metrics['auc']:.4f}")

    if len(results) > 1:
        build_results_table(results)

    if not args.skip_analysis and "iTransformer" in models_to_train:
        run_attention_analysis(test_loader)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*65}")
    print(f"  [OK] Pipeline complete! Total time: {total_elapsed/60:.1f} min")
    print(f"  Results saved to: {RESULT_DIR}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
