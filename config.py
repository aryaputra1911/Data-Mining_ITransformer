"""
Configuration for GEMASTIK iTransformer Training Pipeline.
All constants, paths, column definitions, and hyperparameters.

Deep Optimization v4:
  - N_TEMPORAL=15 context features (calendar + events + key spreads)
  - sqrt(pos_weight) to prevent gradient explosion
  - batch_size=64 for stable AUC-ROC
  - dropout=0.2 for regularization
  - SWA for last 30% of epochs
"""
import math
import torch
from pathlib import Path

# ── Device ──
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Paths ──
BASE_PATH  = Path(r"C:\Users\ARYA\Gemastik div III\Dataset\processed")
TRAIN_FEAT = BASE_PATH / "df_train_features.csv"
VAL_FEAT   = BASE_PATH / "df_val_features.csv"
TEST_FEAT  = BASE_PATH / "df_test_features.csv"
TRAIN_LBL  = BASE_PATH / "df_train_labels.csv"
VAL_LBL    = BASE_PATH / "df_val_labels.csv"
TEST_LBL   = BASE_PATH / "df_test_labels.csv"
LEVEL_CSV  = BASE_PATH / "df_level_full.csv"
ADJ_CSV    = BASE_PATH / "adjacency_matrix_6x6.csv"
SCALERS    = BASE_PATH / "scalers.pkl"
CKPT_DIR   = Path(r"C:\Users\ARYA\Gemastik div III\checkpoints")
RESULT_DIR = Path(r"C:\Users\ARYA\Gemastik div III\results")

# ── Dimensions ──
N_VARIATES      = 33     # core price log-return variates
N_TEMPORAL       = 15    # temporal/context features for iTransformer
N_FEATURES_FULL  = 996
N_LABELS         = 30
LOOKBACK         = 30
HORIZON          = 14

# ── Core variate columns (33 log_return columns — prediction targets) ──
CORE_VARIATES = [
    'beras_dki_cons','beras_jabar_cons','beras_jateng_cons','beras_jatim_cons',
    'beras_sulsel_cons','beras_sumut_cons',
    'bawang_dki_cons','bawang_jabar_cons','bawang_jateng_cons','bawang_jatim_cons',
    'bawang_sulsel_cons','bawang_sumut_cons',
    'cabai_dki_cons','cabai_jabar_cons','cabai_jateng_cons','cabai_jatim_cons',
    'cabai_sulsel_cons','cabai_sumut_cons',
    'beras_jabar_prod','beras_jateng_prod','beras_jatim_prod',
    'beras_sulsel_prod','beras_sumut_prod',
    'bawang_jabar_prod','bawang_jateng_prod','bawang_jatim_prod',
    'bawang_sulsel_prod','bawang_sumut_prod',
    'cabai_jabar_prod','cabai_jateng_prod','cabai_jatim_prod',
    'cabai_sulsel_prod','cabai_sumut_prod',
]
CORE_VARIATE_COLS = [f"log_return_{v}" for v in CORE_VARIATES]

# ── Temporal/Context columns (15 features — global context for iTransformer) ──
TEMPORAL_COLS = [
    # Calendar cyclical (6)
    'dow_sin', 'dow_cos',
    'month_sin', 'month_cos',
    'woy_sin', 'woy_cos',
    # Event flags (4)
    'event_ramadan', 'event_lebaran',
    'event_bbm_2022', 'event_elnino_2023',
    # Key cross-market spread signals (5)
    'spread_pct_beras_jabar', 'spread_pct_bawang_jabar', 'spread_pct_cabai_jabar',
    'spread_vol_beras_jabar', 'spread_vol_cabai_jabar',
]

# ── Spike label columns (30 labels, exact CSV ordering) ──
LABEL_COLS = [
    'spike_beras_dki_cons','spike_beras_jabar_cons','spike_beras_jateng_cons',
    'spike_beras_jatim_cons','spike_beras_sulsel_cons','spike_beras_sumut_cons',
    'spike_bawang_dki_cons','spike_bawang_jabar_cons','spike_bawang_jateng_cons',
    'spike_bawang_jatim_cons','spike_bawang_sulsel_cons','spike_bawang_sumut_cons',
    'spike_cabai_dki_cons','spike_cabai_jabar_cons','spike_cabai_jateng_cons',
    'spike_cabai_jatim_cons','spike_cabai_sulsel_cons','spike_cabai_sumut_cons',
    'spike_beras_jabar_prod','spike_beras_jateng_prod',
    'spike_beras_sulsel_prod','spike_beras_sumut_prod',
    'spike_bawang_jabar_prod','spike_bawang_jateng_prod',
    'spike_bawang_sulsel_prod','spike_bawang_sumut_prod',
    'spike_cabai_jabar_prod','spike_cabai_jateng_prod',
    'spike_cabai_sulsel_prod','spike_cabai_sumut_prod',
]

# ── Pos weights: sqrt(ratio) to prevent gradient explosion ──
_RAW_POS_WEIGHTS = [
    # beras cons (6): dki, jabar, jateng, jatim, sulsel, sumut
    363.5, 120.5, 39.5, 80.0, 55.1, 32.1,
    # bawang cons (6)
    44.6, 21.1, 20.4, 23.3, 39.5, 51.1,
    # cabai cons (6)
    25.0, 19.2, 24.1, 37.4, 35.4, 47.6,
    # beras prod (4): jabar, jateng, sulsel, sumut
    80.0, 51.1, 120.5, 80.0,
    # bawang prod (4)
    44.6, 44.6, 59.7, 47.6,
    # cabai prod (4)
    65.3, 44.6, 35.4, 59.7,
]
POS_WEIGHTS_ORDERED = [round(math.sqrt(w), 2) for w in _RAW_POS_WEIGHTS]

# ── Training hyperparameters ──
BATCH_SIZE      = 64      # v4: larger batches → stable AUC, both classes in each batch
MAX_EPOCHS      = 100
LR              = 1e-4
WEIGHT_DECAY    = 1e-5
PATIENCE        = 20
WARMUP_EPOCHS   = 10
GAMMA_FOCAL     = 2.0
LABEL_SMOOTHING = 0.1
LAMBDA_FORECAST = 1.0
LAMBDA_SPIKE    = 0.5

# ── iTransformer architecture ──
ITF_D_MODEL     = 64
ITF_N_HEADS     = 4
ITF_N_LAYERS    = 2
ITF_D_FF        = 128
ITF_DROPOUT     = 0.2
ITF_ATTN_TEMP   = 0.5

# ── Baseline architecture (downsized for fair comparison ~100k params) ──
BASE_D_MODEL    = 64
BASE_N_HEADS    = 4
BASE_N_LAYERS   = 1     # single layer to prevent overfitting on <1k samples
BASE_DROPOUT    = 0.3   # heavier dropout for baselines

# ── Regularization ──
GAUSSIAN_NOISE_STD = 0.01
SWA_START_EPOCH    = 70
SWA_LR             = 1e-5

