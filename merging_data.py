import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# PATH
# ─────────────────────────────────────────────────────────────
BASE_PATH       = Path(r"C:\Users\ARYA\Gemastik div III\Dataset\raw")
OUT_PATH        = Path(r"C:\Users\ARYA\Gemastik div III\Dataset\processed")
FOLDER_KONSUMEN = BASE_PATH / "Pasar_tradisional_Konsumen"
FOLDER_PRODUSEN = BASE_PATH / "Produsen"

# ─────────────────────────────────────────────────────────────
# NAMA FILE — hardcoded persis sesuai direktori
# ─────────────────────────────────────────────────────────────
FILES_KONSUMEN = {
    "dki": [
        "DKI Jakarta_2021_pasar_tradisional.xlsx",
        "DKI Jakarta_2022_pasar_tradisional.xlsx",
        "DKI Jakarta_2023_pasar_tradisional.xlsx",
        "DKI Jakarta_2024_pasar_tradisional.xlsx",
        "DKI Jakarta_2025_pasar_tradisional.xlsx",
    ],
    "jabar": [
        "Jawa Barat_2021_Pasar_tradisional.xlsx",
        "Jawa Barat_2022_Pasar_tradisional.xlsx",
        "Jawa Barat_2023_Pasar_tradisional.xlsx",
        "Jawa Barat_2024_Pasar_tradisional.xlsx",
        "Jawa Barat_2025_Pasar_tradisional.xlsx",
    ],
    "jateng": [
        "Jawa Tengah_2021_Pasar_tradisional.xlsx",
        "Jawa Tengah_2022_Pasar_tradisional.xlsx",
        "Jawa Tengah_2023_Pasar_tradisional.xlsx",
        "Jawa Tengah_2024_Pasar_tradisional.xlsx",
        "Jawa Tengah_2025_Pasar_tradisional.xlsx",
    ],
    "jatim": [
        "Jawa Timur_2021_Pasar_tradisional.xlsx",
        "Jawa Timur_2022_Pasar_tradisional.xlsx",
        "Jawa Timur_2023_Pasar_tradisional.xlsx",
        "Jawa Timur_2024_Pasar_tradisional.xlsx",
        "Jawa Timur_2025_Pasar_tradisional.xlsx",
    ],
    "sulsel": [
        "Sulsel_2021_Pasar_tradisional.xlsx",
        "Sulsel_2022_pasar_tradisional.xlsx",
        "Sulsel_2023_Pasar_tradisional.xlsx",
        "Sulsel_2024_Pasar_tradisional.xlsx",  
        "Sulsel_2025_Pasar_tradisional.xlsx",
    ],
    "sumut": [
        "Sumut_2021_Pasar_tradisional.xlsx",
        "Sumut_2022_Pasar_tradisional.xlsx",
        "Sumut_2023_Pasar_tradisional.xlsx",
        "Sumut_2024_Pasar_tradisional.xlsx",
        "Sumut_2025_Pasar_tradisional.xlsx",
    ],
}

FILES_PRODUSEN = {
    "jabar":  ["Jabar_2021_Produsen.xlsx",  "Jabar_2022_Produsen.xlsx",
               "Jabar_2023_Produsen.xlsx",  "Jabar_2024_Produsen.xlsx",
               "Jabar_2025_Produsen.xlsx"],
    "jateng": ["Jateng_2021_Produsen.xlsx", "Jateng_2022_Produsen.xlsx",
               "Jateng_2023_Produsen.xlsx", "Jateng_2024_Produsen.xlsx",
               "Jateng_2025_Produsen.xlsx"],
    "jatim":  ["Jatim_2021_Produsen.xlsx",  "Jatim_2022_Produsen.xlsx",
               "Jatim_2023_Produsen.xlsx",  "Jatim_2024_Produsen.xlsx",
               "Jatim_2025_Produsen.xlsx"],
    "sulsel": ["Sulsel_2021_Produsen.xlsx", "Sulsel_2022_Produsen.xlsx",
               "Sulsel_2023_Produsen.xlsx", "Sulsel_2024_Produsen.xlsx",
               "Sulsel_2025_Produsen.xlsx"],
    "sumut":  ["Sumut_2021_Produsen.xlsx",  "Sumut_2022_Produsen.xlsx",
               "Sumut_2023_Produsen.xlsx",  "Sumut_2024_Produsen.xlsx",
               "Sumut_2025_Produsen.xlsx"],
}

# ─────────────────────────────────────────────────────────────
# KOMODITAS — diverifikasi dari kedua file
# ─────────────────────────────────────────────────────────────
KOMODITAS_MAP = {
    "Beras Kualitas Medium I"    : "beras",
    "Beras Kualitas Medium II"   : "beras",
    "Bawang Merah Ukuran Sedang" : "bawang",
    "Cabai Merah Keriting"       : "cabai",
    "Cabai Merah Keriting "      : "cabai",  # trailing space — ada di kedua file
}

KOLOM_KONSUMEN = (
    [f"beras_{p}"  for p in ["dki","jabar","jateng","jatim","sulsel","sumut"]] +
    [f"bawang_{p}" for p in ["dki","jabar","jateng","jatim","sulsel","sumut"]] +
    [f"cabai_{p}"  for p in ["dki","jabar","jateng","jatim","sulsel","sumut"]]
)
KOLOM_PRODUSEN = (
    [f"beras_{p}"  for p in ["jabar","jateng","jatim","sulsel","sumut"]] +
    [f"bawang_{p}" for p in ["jabar","jateng","jatim","sulsel","sumut"]] +
    [f"cabai_{p}"  for p in ["jabar","jateng","jatim","sulsel","sumut"]]
)

# ─────────────────────────────────────────────────────────────
# FUNGSI 1 — Baca satu file
# ─────────────────────────────────────────────────────────────
def read_one_file(filepath: Path, provinsi: str) -> pd.DataFrame:
    if not filepath.exists():
        print(f"    ⚠️  TIDAK ADA : {filepath.name}")
        return pd.DataFrame()
    try:
        df_raw = pd.read_excel(filepath, header=0, dtype=str)
    except Exception as e:
        print(f"    ❌ GAGAL BACA: {filepath.name} | {e}")
        return pd.DataFrame()

    date_cols = df_raw.columns[2:]
    records   = []

    for _, row in df_raw.iterrows():
        nama_raw = str(row.iloc[1]).strip()
        if nama_raw not in KOMODITAS_MAP:
            continue
        komoditas = KOMODITAS_MAP[nama_raw]

        for date_str in date_cols:
            date_clean = str(date_str).replace(" ", "")
            try:
                date_parsed = pd.to_datetime(date_clean, format="%d/%m/%Y")
            except Exception:
                continue

            val = str(row[date_str]).strip()
            if val in ["-", "nan", "None", ""]:
                harga = np.nan
            else:
                try:
                    harga = float(val.replace(",", ""))
                except Exception:
                    harga = np.nan

            records.append({
                "date"     : date_parsed,
                "komoditas": komoditas,
                "harga"    : harga,
                "provinsi" : provinsi,
            })

    if not records:
        print(f"    ⚠️  Tidak ada komoditas target di: {filepath.name}")
        return pd.DataFrame()

    return pd.DataFrame(records)

# ─────────────────────────────────────────────────────────────
# FUNGSI 2 — Build dataset per tipe
# ─────────────────────────────────────────────────────────────
def build_dataset(tipe: str) -> pd.DataFrame:
    if tipe == "konsumen":
        folder, files_map, kolom_urut = FOLDER_KONSUMEN, FILES_KONSUMEN, KOLOM_KONSUMEN
    else:
        folder, files_map, kolom_urut = FOLDER_PRODUSEN, FILES_PRODUSEN, KOLOM_PRODUSEN

    semua_long, n_ok, n_gagal, gagal_list = [], 0, 0, []

    for provinsi, file_list in files_map.items():
        for fname in file_list:
            print(f"  {fname}")
            df = read_one_file(folder / fname, provinsi)
            if df.empty:
                n_gagal += 1
                gagal_list.append(fname)
            else:
                n_ok += 1
                print(f"    ✅ {len(df):,} records")
                semua_long.append(df)

    print(f"\n  Hasil: {n_ok} OK | {n_gagal} gagal")
    if gagal_list:
        for f in gagal_list:
            print(f"    ❌ {f}")

    if not semua_long:
        return pd.DataFrame()

    df_all = pd.concat(semua_long, ignore_index=True)

    # Handle duplikat
    dup = df_all.duplicated(subset=["date","komoditas","provinsi"], keep=False)
    if dup.any():
        print(f"  ⚠️  {dup.sum()} duplikat → rata-rata")
        df_all = df_all.groupby(
            ["date","komoditas","provinsi"], as_index=False
        )["harga"].mean()

    df_all["col_name"] = df_all["komoditas"] + "_" + df_all["provinsi"]

    # Pivot wide
    df_wide = df_all.pivot_table(
        index="date", columns="col_name",
        values="harga", aggfunc="mean"
    )
    df_wide.columns.name = None
    df_wide.index.name   = "date"

    # Reindex kalender harian penuh — NaN = weekend/libur
    kalender = pd.date_range("2021-01-01", "2025-12-31", freq="D")
    df_wide  = df_wide.reindex(kalender)
    df_wide.index.name = "date"

    # Urutkan kolom
    kolom_ada    = [c for c in kolom_urut if c in df_wide.columns]
    kolom_ekstra = [c for c in df_wide.columns if c not in kolom_urut]
    kolom_hilang = [c for c in kolom_urut if c not in df_wide.columns]
    if kolom_ekstra:  print(f"  ⚠️  Kolom ekstra: {kolom_ekstra}")
    if kolom_hilang:  print(f"  ❌ Kolom hilang: {kolom_hilang}")

    return df_wide[kolom_ada + kolom_ekstra]

# ─────────────────────────────────────────────────────────────
# FUNGSI 3 — Summary
# ─────────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame, nama: str):
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {nama}")
    print(f"  Shape  : {df.shape[0]} baris x {df.shape[1]} kolom")
    print(f"  Periode: {df.index.min().date()} -> {df.index.max().date()}")
    print(f"  Kolom  : {list(df.columns)}")
    print(f"\n  % Missing per kolom (NaN = weekend/libur, belum diinterpolasi):")
    pct = (df.isna().mean() * 100).round(1)
    for col, p in pct.items():
        bar = "=" * int(p / 5)
        print(f"    {col:<25} {p:>5.1f}% |{bar}")
    print(f"\n  Statistik harga (Rp):")
    print(df.describe().round(0).to_string())

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  BUILD DATASET — GEMASTIK iTransformer")
    print("=" * 60)
    print(f"\n  Konsumen folder : {FOLDER_KONSUMEN}")
    print(f"  Produsen folder : {FOLDER_PRODUSEN}")
    print(f"  Output          : {OUT_PATH}")

    print("\n" + "!"*60)
    print("  PASTIKAN sudah rename 4 file Sulsel konsumen:")
    print("  Sulsel_202X_Pasar Tradisional.xlsx.xlsx")
    print("  -> Sulsel_202X_Pasar Tradisional.xlsx")
    print("  (untuk tahun 2022, 2023, 2024, 2025)")
    print("!"*60)
    input("\n  Tekan ENTER jika sudah siap...\n")

    # Konsumen
    print("\n" + "="*60)
    print("  [1/2] KONSUMEN (pasar tradisional)")
    print("="*60)
    df_k = build_dataset("konsumen")
    if not df_k.empty:
        path_k = OUT_PATH / "konsumen_raw.csv"
        df_k.to_csv(path_k)
        print(f"\n  Tersimpan: {path_k}")
        print_summary(df_k, "konsumen_raw.csv")

    # Produsen
    print("\n" + "="*60)
    print("  [2/2] PRODUSEN")
    print("="*60)
    df_p = build_dataset("produsen")
    if not df_p.empty:
        path_p = OUT_PATH / "produsen_raw.csv"
        df_p.to_csv(path_p)
        print(f"\n  Tersimpan: {path_p}")
        print_summary(df_p, "produsen_raw.csv")

    print("\n" + "="*60)
    print("  SELESAI — Lanjut ke EDA")
    print("="*60)

if __name__ == "__main__":
    main()