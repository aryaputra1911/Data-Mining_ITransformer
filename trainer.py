"""
Training and evaluation loop for GEMASTIK iTransformer Pipeline.

v4 Deep Optimization:
  - Passes x_temporal to models (indices 33:48)
  - AUPRC as primary spike metric
  - SWA for last 30% of epochs
  - Threshold optimization on val set
"""
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
)
from torch.optim.swa_utils import AveragedModel, SWALR

from config import (
    DEVICE, N_VARIATES, N_TEMPORAL, MAX_EPOCHS, LR, WEIGHT_DECAY, PATIENCE,
    WARMUP_EPOCHS, CKPT_DIR, RESULT_DIR, CORE_VARIATES,
    SWA_START_EPOCH, SWA_LR,
)


# =============================================================================
# Metrics
# =============================================================================

def smape(y_pred, y_true, eps=1e-8):
    """Symmetric Mean Absolute Percentage Error -- safe for near-zero values."""
    return float(200.0 * np.mean(
        np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + eps)
    ))


def find_optimal_threshold(spike_true, spike_pred_prob, thresholds=None):
    """Find optimal threshold maximizing macro F1 on validation set.
    Searches 0.01-0.99 for fair, model-independent threshold selection."""
    if thresholds is None:
        thresholds = np.arange(0.01, 0.99, 0.01)

    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        binary = (spike_pred_prob > t).astype(int)
        f1 = f1_score(spike_true, binary, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    return best_t, best_f1


# =============================================================================
# Training
# =============================================================================

def _split_input(x, n_variates=N_VARIATES, n_temporal=N_TEMPORAL):
    """Split full feature tensor into price and temporal components."""
    x_price    = x[:, :, :n_variates]                              # (B, T, 33)
    x_temporal = x[:, :, n_variates:n_variates + n_temporal]       # (B, T, 15)
    return x_price, x_temporal


def train_one_epoch(model, loader, optimizer, criterion, device, epoch=0):
    """Train for one epoch, return avg losses."""
    model.train()
    total_loss, total_fc, total_sp = 0.0, 0.0, 0.0
    spike_prob_sum, spike_count = 0.0, 0

    for batch in loader:
        x     = batch["x"].to(device)
        y     = batch["y"].to(device)
        spike = batch["spike"].to(device)

        # v4: Split into price + temporal
        x_price, x_temporal = _split_input(x)

        optimizer.zero_grad()
        y_pred, spike_pred = model(x_price, x_temporal)
        loss, lfc, lsp = criterion(y_pred, y, spike_pred, spike)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_fc   += lfc
        total_sp   += lsp

        with torch.no_grad():
            spike_prob_sum += torch.sigmoid(spike_pred).mean().item()
            spike_count += 1

    n = len(loader)
    avg_spike_prob = spike_prob_sum / max(spike_count, 1)

    if epoch <= 1 or epoch % 10 == 0:
        print(f"    [debug] avg spike_prob={avg_spike_prob:.4f} | "
              f"focal_loss={total_sp/n:.4f} | forecast_loss={total_fc/n:.4f}")

    return total_loss / n, total_fc / n, total_sp / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    """Evaluate model -- v4: includes AUPRC as primary spike metric."""
    model.eval()
    total_loss = 0.0
    all_spike_pred, all_spike_true = [], []
    all_y_pred, all_y_true = [], []

    for batch in loader:
        x     = batch["x"].to(device)
        y     = batch["y"].to(device)
        spike = batch["spike"].to(device)

        x_price, x_temporal = _split_input(x)
        y_pred, spike_pred = model(x_price, x_temporal)
        loss, _, _ = criterion(y_pred, y, spike_pred, spike)
        total_loss += loss.item()

        all_spike_pred.append(torch.sigmoid(spike_pred).cpu())
        all_spike_true.append(spike.cpu())
        all_y_pred.append(y_pred.cpu())
        all_y_true.append(y.cpu())

    spike_pred_all = torch.cat(all_spike_pred).numpy()
    spike_true_all = torch.cat(all_spike_true).numpy()
    y_pred_all     = torch.cat(all_y_pred).numpy()
    y_true_all     = torch.cat(all_y_true).numpy()

    # Regression metrics
    mae   = float(np.mean(np.abs(y_pred_all - y_true_all)))
    rmse  = float(np.sqrt(np.mean((y_pred_all - y_true_all) ** 2)))
    smape_val = smape(y_pred_all, y_true_all)

    # Per-variate MAE
    mae_per_var = np.mean(np.abs(y_pred_all - y_true_all), axis=(0, 1))

    # Classification metrics
    spike_binary = (spike_pred_all > threshold).astype(int)
    f1   = f1_score(spike_true_all, spike_binary, average='macro', zero_division=0)
    prec = precision_score(spike_true_all, spike_binary, average='macro', zero_division=0)
    rec  = recall_score(spike_true_all, spike_binary, average='macro', zero_division=0)

    # AUC-ROC (per-column fallback for sparse labels)
    try:
        auc = roc_auc_score(spike_true_all, spike_pred_all, average='macro')
    except ValueError:
        aucs = []
        for col in range(spike_true_all.shape[1]):
            if spike_true_all[:, col].sum() > 0:
                try:
                    aucs.append(roc_auc_score(spike_true_all[:, col], spike_pred_all[:, col]))
                except ValueError:
                    pass
        auc = float(np.mean(aucs)) if aucs else float('nan')

    # v4: AUPRC -- primary metric for imbalanced spike detection
    try:
        auprc = average_precision_score(spike_true_all, spike_pred_all, average='macro')
    except ValueError:
        auprcs = []
        for col in range(spike_true_all.shape[1]):
            if spike_true_all[:, col].sum() > 0:
                try:
                    auprcs.append(average_precision_score(
                        spike_true_all[:, col], spike_pred_all[:, col]
                    ))
                except ValueError:
                    pass
        auprc = float(np.mean(auprcs)) if auprcs else float('nan')

    return {
        "loss": total_loss / len(loader),
        "mae": mae, "rmse": rmse, "smape": smape_val,
        "f1": f1, "precision": prec, "recall": rec,
        "auc": auc, "auprc": auprc,
        "threshold": threshold,
        "mae_per_var": mae_per_var,
        "spike_pred_all": spike_pred_all,
        "spike_true_all": spike_true_all,
    }


def evaluate_with_threshold_search(model, val_loader, test_loader, criterion, device):
    """Find optimal threshold on val, evaluate test with that threshold."""
    val_metrics = evaluate(model, val_loader, criterion, device, threshold=0.5)

    opt_t, opt_f1 = find_optimal_threshold(
        val_metrics['spike_true_all'], val_metrics['spike_pred_all']
    )
    print(f"    Optimal threshold: {opt_t:.2f} (val F1={opt_f1:.4f})")

    val_metrics = evaluate(model, val_loader, criterion, device, threshold=opt_t)
    test_metrics = evaluate(model, test_loader, criterion, device, threshold=opt_t)

    return val_metrics, test_metrics, opt_t


# =============================================================================
# Main training loop
# =============================================================================

def train_model(model, model_name, train_loader, val_loader, criterion,
                max_epochs=MAX_EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
                patience=PATIENCE, warmup_epochs=WARMUP_EPOCHS, device=DEVICE,
                swa_start=SWA_START_EPOCH, swa_lr=SWA_LR):
    """
    Full training loop with warmup, ReduceLROnPlateau, SWA, early stopping.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs,
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7,
    )

    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
    swa_active = False

    best_val_loss = float('inf')
    patience_counter = 0
    history = []

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*65}")
    print(f"  Training: {model_name}")
    print(f"  Parameters: {n_params:,}")
    print(f"  LR: {lr} | Warmup: {warmup_epochs} ep | SWA from ep {swa_start}")
    print(f"  Device: {device}")
    print(f"{'='*65}")

    for epoch in range(1, max_epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        tr_loss, tr_fc, tr_sp = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        val_metrics = evaluate(model, val_loader, criterion, device, threshold=0.5)

        # Scheduler: warmup -> plateau -> SWA
        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        elif epoch >= swa_start:
            if not swa_active:
                print(f"  [SWA] Activating Stochastic Weight Averaging at epoch {epoch}")
                swa_active = True
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            plateau_scheduler.step(val_metrics['loss'])

        history.append({
            "epoch": epoch, "lr": current_lr,
            "train_loss": tr_loss, "train_forecast": tr_fc, "train_spike": tr_sp,
            "val_loss": val_metrics['loss'],
            "val_mae": val_metrics['mae'], "val_rmse": val_metrics['rmse'],
            "val_smape": val_metrics['smape'],
            "val_f1": val_metrics['f1'], "val_precision": val_metrics['precision'],
            "val_recall": val_metrics['recall'], "val_auc": val_metrics['auc'],
            "val_auprc": val_metrics['auprc'],
            "swa_active": swa_active,
        })

        if epoch % 5 == 0 or epoch == 1:
            swa_tag = " [SWA]" if swa_active else ""
            print(f"  Ep {epoch:3d} | lr={current_lr:.2e}{swa_tag} | "
                  f"tr={tr_loss:.4f} (fc={tr_fc:.4f} sp={tr_sp:.4f}) | "
                  f"val={val_metrics['loss']:.4f} | "
                  f"mae={val_metrics['mae']:.4f} | "
                  f"f1={val_metrics['f1']:.4f} | "
                  f"auprc={val_metrics['auprc']:.4f}")

        # Early stopping (disabled during SWA)
        if not swa_active:
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                CKPT_DIR.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'val_metrics': {k: v for k, v in val_metrics.items()
                                    if k not in ('mae_per_var', 'spike_pred_all', 'spike_true_all')},
                }, CKPT_DIR / f"{model_name}_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  [STOP] Early stopping at epoch {epoch}")
                    break

    # SWA finalization
    if swa_active:
        print("  [SWA] Updating batch normalization statistics...")
        swa_model.train()
        with torch.no_grad():
            for batch in train_loader:
                x = batch["x"].to(device)
                x_price, x_temporal = _split_input(x)
                swa_model(x_price, x_temporal)

        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state': swa_model.module.state_dict(),
            'val_loss': val_metrics['loss'],
            'swa': True,
        }, CKPT_DIR / f"{model_name}_swa.pt")
        print(f"  [SWA] Saved SWA checkpoint: {model_name}_swa.pt")
        model.load_state_dict(swa_model.module.state_dict())

    # Save history
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(RESULT_DIR / f"{model_name}_history.csv", index=False)
    print(f"  Best val loss: {best_val_loss:.4f}")

    return model, history


def save_per_variate_mae(mae_per_var, model_name, variate_names=None):
    """Save per-variate MAE breakdown for paper table."""
    if variate_names is None:
        variate_names = CORE_VARIATES
    df = pd.DataFrame({
        "variate": variate_names,
        "mae": [f"{v:.6f}" for v in mae_per_var],
    })
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULT_DIR / f"{model_name}_mae_per_variate.csv", index=False)
    print(f"  [OK] Saved per-variate MAE: {RESULT_DIR / f'{model_name}_mae_per_variate.csv'}")
    return df


# =============================================================================
# Fix 1: Price-space evaluation (inverse transform to absolute prices)
# =============================================================================

def evaluate_price_space(model, test_ds, device, model_name="model"):
    """
    Evaluate in absolute price space by inverse-transforming predictions.

    Steps:
      1. Inverse StandardScaler: raw_return = scaled * std + mean
      2. Reconstruct price: P_{t+h} = P_{t+h-1} * exp(raw_return / 100)
      3. Compute MAE, RMSE, sMAPE on absolute prices (Rp)
    """
    import joblib
    from config import SCALERS, LEVEL_CSV, CORE_VARIATE_COLS, TEST_FEAT, N_TEMPORAL

    # Load scalers and level prices
    scaler_dict = joblib.load(SCALERS)
    df_level = pd.read_csv(LEVEL_CSV, index_col='date', parse_dates=True)
    test_dates = pd.to_datetime(pd.read_csv(TEST_FEAT)['date'])

    # Extract scaler params for 33 core variates (some may be missing if zero-filled)
    means, stds = [], []
    for c in CORE_VARIATE_COLS:
        if c in scaler_dict:
            means.append(scaler_dict[c].mean_[0])
            stds.append(scaler_dict[c].scale_[0])
        else:
            # Missing scaler = column was zero-filled, identity transform
            means.append(0.0)
            stds.append(1.0)
    means = np.array(means)
    stds  = np.array(stds)
    level_cols = [c.replace('log_return_', '') for c in CORE_VARIATE_COLS]

    lookback = test_ds.lookback
    horizon  = test_ds.horizon
    n_var    = len(CORE_VARIATE_COLS)

    model.eval()
    pred_prices_list = []
    true_prices_list = []

    with torch.no_grad():
        for idx in range(len(test_ds)):
            sample = test_ds[idx]
            x = sample['x'].unsqueeze(0).to(device)
            y_true_scaled = sample['y'].numpy()  # (horizon, 33)

            x_price    = x[:, :, :n_var]
            x_temporal = x[:, :, n_var:n_var + N_TEMPORAL]
            y_pred_scaled, _ = model(x_price, x_temporal)
            y_pred_scaled = y_pred_scaled[0].cpu().numpy()  # (horizon, 33)

            # Inverse scale to raw log-returns
            y_pred_raw = y_pred_scaled * stds + means
            y_true_raw = y_true_scaled * stds + means

            # Get last known absolute price before this prediction window
            last_idx = idx + lookback - 1
            if last_idx >= len(test_dates):
                continue
            last_date = test_dates.iloc[last_idx]

            # Find closest date in level CSV
            if last_date in df_level.index:
                last_price = df_level.loc[last_date, level_cols].values.astype(float)
            else:
                # Find nearest prior date
                prior = df_level.index[df_level.index <= last_date]
                if len(prior) == 0:
                    continue
                last_price = df_level.loc[prior[-1], level_cols].values.astype(float)

            if np.any(np.isnan(last_price)):
                continue

            # Reconstruct absolute prices step-by-step
            pred_p = np.zeros((horizon, n_var))
            true_p = np.zeros((horizon, n_var))
            prev_pred = last_price.copy()
            prev_true = last_price.copy()

            for h in range(horizon):
                pred_p[h] = prev_pred * np.exp(y_pred_raw[h] / 100.0)
                true_p[h] = prev_true * np.exp(y_true_raw[h] / 100.0)
                prev_pred = pred_p[h]
                prev_true = true_p[h]

            pred_prices_list.append(pred_p)
            true_prices_list.append(true_p)

    if not pred_prices_list:
        print("  [WARN] No valid windows for price-space evaluation.")
        return {}

    y_pred_arr = np.stack(pred_prices_list)
    y_true_arr = np.stack(true_prices_list)

    # Price-space metrics
    price_mae  = float(np.mean(np.abs(y_pred_arr - y_true_arr)))
    price_rmse = float(np.sqrt(np.mean((y_pred_arr - y_true_arr) ** 2)))
    price_smape = float(200.0 * np.mean(
        np.abs(y_pred_arr - y_true_arr) /
        (np.abs(y_pred_arr) + np.abs(y_true_arr) + 1e-8)
    ))

    # Per-variate price MAE
    var_mae = np.mean(np.abs(y_pred_arr - y_true_arr), axis=(0, 1))

    results = {
        "price_mae": price_mae,
        "price_rmse": price_rmse,
        "price_smape": price_smape,
        "price_mae_per_var": var_mae,
    }

    print(f"\n  PRICE-SPACE METRICS ({model_name}):")
    print(f"    MAE  : Rp {price_mae:,.0f}")
    print(f"    RMSE : Rp {price_rmse:,.0f}")
    print(f"    sMAPE: {price_smape:.2f}%")

    # Save per-variate breakdown
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    price_df = pd.DataFrame({
        "variate": level_cols,
        "price_mae_rp": [f"{v:,.0f}" for v in var_mae],
    })
    price_df.to_csv(RESULT_DIR / f"{model_name}_price_mae.csv", index=False)

    return results
