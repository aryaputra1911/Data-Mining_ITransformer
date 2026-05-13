"""
Dataset and DataLoader for GEMASTIK iTransformer Pipeline.
Sliding-window dataset shared by all 4 models.

v4: Column ordering = [33 core variates | 15 temporal | rest]
    so x[:,:,:33] = price, x[:,:,33:48] = temporal context.
"""
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from config import (
    TRAIN_FEAT, VAL_FEAT, TEST_FEAT,
    TRAIN_LBL, VAL_LBL, TEST_LBL,
    CORE_VARIATE_COLS, TEMPORAL_COLS, LABEL_COLS,
    LOOKBACK, HORIZON, BATCH_SIZE, N_VARIATES, N_TEMPORAL,
)


class FoodPriceDataset(Dataset):
    """
    Sliding window dataset for multivariate time-series forecasting + spike classification.

    Column ordering: [33 core variates | 15 temporal | remaining features]
    This ensures:
      x[:, :, :33]    → price log-returns (model input for all models)
      x[:, :, 33:48]  → temporal/context features (iTransformer only)

    For each window index `idx`:
      - x    : (lookback, n_features)   — full feature input (ordered)
      - y    : (horizon,  33)           — log_return of 33 core variates (regression target)
      - spike: (30,)                    — binary spike labels (classification target)
    """

    def __init__(self, df_features: pd.DataFrame, df_labels: pd.DataFrame,
                 lookback: int = LOOKBACK, horizon: int = HORIZON,
                 core_variate_cols: list = None, temporal_cols: list = None,
                 feature_cols: list = None, label_cols: list = None):

        if core_variate_cols is None:
            core_variate_cols = CORE_VARIATE_COLS
        if temporal_cols is None:
            temporal_cols = TEMPORAL_COLS
        if label_cols is None:
            label_cols = LABEL_COLS
        if feature_cols is None:
            feature_cols = [c for c in df_features.columns if c != 'date']

        self.lookback = lookback
        self.horizon  = horizon

        # v4: Column ordering — core variates first, then temporal, then rest
        # Filter to only columns that actually exist in the dataframe
        available_temporal = [c for c in temporal_cols if c in df_features.columns]
        if len(available_temporal) < len(temporal_cols):
            missing = set(temporal_cols) - set(available_temporal)
            print(f"  [WARN] Missing temporal columns (will be zero-filled): {missing}")

        # Build ordered column list: [core | temporal | other]
        used = set(core_variate_cols) | set(available_temporal)
        other_cols = [c for c in feature_cols if c not in used]
        ordered_cols = core_variate_cols + available_temporal + other_cols

        # Zero-fill any missing temporal columns
        for c in temporal_cols:
            if c not in df_features.columns:
                df_features = df_features.copy()
                df_features[c] = 0.0

        # Rebuild with full temporal cols in correct order
        used_full = set(core_variate_cols) | set(temporal_cols)
        other_cols_full = [c for c in feature_cols if c not in used_full]
        ordered_cols = core_variate_cols + temporal_cols + other_cols_full

        self.X      = torch.tensor(df_features[ordered_cols].values, dtype=torch.float32)
        self.y      = torch.tensor(df_features[core_variate_cols].values, dtype=torch.float32)
        self.labels = torch.tensor(df_labels[label_cols].values, dtype=torch.float32)

        self.n_features   = len(ordered_cols)
        self.n_variates   = len(core_variate_cols)
        self.n_temporal   = len(temporal_cols)
        self.n_labels     = len(label_cols)
        self.feature_cols = ordered_cols

    def __len__(self):
        return len(self.X) - self.lookback - self.horizon + 1

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.lookback]                                # (30, n_features)
        y_seq = self.y[idx + self.lookback : idx + self.lookback + self.horizon]  # (14, 33)
        # Spike label at end of lookback window (already forward-looking)
        spike = self.labels[idx + self.lookback - 1]                              # (30,)
        return {"x": x_seq, "y": y_seq, "spike": spike}


def load_datasets():
    """Load all CSV files and create Dataset objects."""
    print("[Data] Loading CSVs...")
    df_train_feat = pd.read_csv(TRAIN_FEAT)
    df_val_feat   = pd.read_csv(VAL_FEAT)
    df_test_feat  = pd.read_csv(TEST_FEAT)
    df_train_lbl  = pd.read_csv(TRAIN_LBL)
    df_val_lbl    = pd.read_csv(VAL_LBL)
    df_test_lbl   = pd.read_csv(TEST_LBL)

    print(f"  Train features: {df_train_feat.shape}, labels: {df_train_lbl.shape}")
    print(f"  Val   features: {df_val_feat.shape},   labels: {df_val_lbl.shape}")
    print(f"  Test  features: {df_test_feat.shape},  labels: {df_test_lbl.shape}")

    train_ds = FoodPriceDataset(df_train_feat, df_train_lbl)
    val_ds   = FoodPriceDataset(df_val_feat,   df_val_lbl)
    test_ds  = FoodPriceDataset(df_test_feat,  df_test_lbl)

    print(f"  Train windows: {len(train_ds)}")
    print(f"  Val   windows: {len(val_ds)}")
    print(f"  Test  windows: {len(test_ds)}")
    print(f"  Column order: [{N_VARIATES} price | {N_TEMPORAL} temporal | rest]")

    return train_ds, val_ds, test_ds


def make_loaders(train_ds, val_ds, test_ds, batch_size=BATCH_SIZE):
    """Create DataLoaders."""
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader
