import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data():
    folder_path = 'TelcoCustomerChurn_raw'

    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' tidak ditemukan.")
        return

    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    if not files:
        print("Error: Tidak ada file CSV di folder TelcoCustomerChurn_raw.")
        return

    input_file = os.path.join(folder_path, files[0])
    df = pd.read_csv(input_file)
    print(f"Memproses data dari: {input_file}")
    print(f"Ukuran Awal: {df.shape}")

    df.drop_duplicates(inplace=True)

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    for col in categorical_cols:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])

    for col in numeric_cols:
        if df[col].nunique() <= 2:
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    target_bin_col = None
    for col in numeric_cols:
        if df[col].nunique() > 5:
            target_bin_col = col
            break

    if target_bin_col:
        new_bin_col = f"{target_bin_col}_category"
        df[new_bin_col] = pd.qcut(df[target_bin_col], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
        categorical_cols.append(new_bin_col)
    else:
        print("Tidak ditemukan kolom numerik yang cocok untuk binning.")

    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    output_folder = 'preprocessing/TelcoCustomerChurn_raw_preprocessing'
    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, 'data_clean.csv')
    df.to_csv(output_path, index=False)

    print("\nProses Selesai! Data berhasil disimpan.")
    print(df.head())

if __name__ == "__main__":
    preprocess_data()
