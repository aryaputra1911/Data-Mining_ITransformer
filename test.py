import pandas as pd
train = pd.read_csv(r"C:\Users\ARYA\Gemastik div III\Dataset\processed\df_train_features.csv", index_col=0)
val   = pd.read_csv(r"C:\Users\ARYA\Gemastik div III\Dataset\processed\df_val_features.csv",   index_col=0)
test  = pd.read_csv(r"C:\Users\ARYA\Gemastik div III\Dataset\processed\df_test_features.csv",  index_col=0)
print("Train:", train.shape)
print("Val  :", val.shape)
print("Test :", test.shape)
print("\nKolom:", list(train.columns))