import sys

import numpy as np
import pandas as pd
from tkinter.filedialog import askopenfilename

from sklearn.metrics import r2_score, mean_absolute_percentage_error,mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def select_file():
    try:
        file_path = askopenfilename(
            title="Selectează un fișier CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not file_path:
            print("Operație anulată de utilizator.")
            sys.exit(0)

        data = pd.read_csv(file_path)
        print("\nDate încărcate cu succes! Primele 5 înregistrări:")
        print(data.head())
        return data

    except Exception as e:
        print(f"Eroare la încărcarea fișierului: {str(e)}")
        sys.exit(1)

def select_data_prestabilite(data):
    input_columns = ["B365H","B365D","B365A"]
    output_columns = ["FTHG","FTAG"]
    return input_columns, output_columns


def select_data(data):
    input_columns = []
    output_columns = []
    try:
        print("\nIntrodu coloanele pentru INPUT (tastează '...' pentru a termina):")
        while True:
            col = input("Coloană input: ").strip()
            if col == "...":
                break
            if col in data.columns:
                input_columns.append(col)
            else:
                print(f"Avertisment: Coloana '{col}' nu există. Încearcă din nou.")

        print("\nIntrodu coloanele pentru OUTPUT (tastează '...' pentru a termina):")
        while True:
            col = input("Coloană output: ").strip()
            if col == "...":
                break
            if col in data.columns:
                output_columns.append(col)
            else:
                print(f"Avertisment: Coloana '{col}' nu există. Încearcă din nou.")

        return input_columns, output_columns

    except KeyboardInterrupt:
        print("\nOperație anulată de utilizator.")
        sys.exit(0)


def normalize_min_max(train_input, train_output, test_input, test_output):
    input_scaler = MinMaxScaler()
    output_scaler = MinMaxScaler()

    train_input_scaled = input_scaler.fit_transform(train_input)
    train_output_scaled = output_scaler.fit_transform(train_output)

    test_input_scaled = input_scaler.transform(test_input)
    test_output_scaled = output_scaler.transform(test_output)

    return train_input_scaled, train_output_scaled, test_input_scaled, test_output_scaled


def normalize_standard(train_input, train_output, test_input, test_output):
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()

    train_input_scaled = input_scaler.fit_transform(train_input)
    train_output_scaled = output_scaler.fit_transform(train_output)

    test_input_scaled = input_scaler.transform(test_input)
    test_output_scaled = output_scaler.transform(test_output)

    return train_input_scaled, train_output_scaled, test_input_scaled, test_output_scaled


def shuffle_division(data, train_percent, input_columns, output_columns):
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    train_size = int(train_percent / 100 * len(data))
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    # extraction
    train_input = train_data[input_columns].values.astype(np.float32)
    train_output = train_data[output_columns].values.astype(np.float32)
    test_input = test_data[input_columns].values.astype(np.float32)
    test_output = test_data[output_columns].values.astype(np.float32)

    return train_input, train_output, test_input, test_output

def calculate_metrics(test_output,predictions):
    mse = mean_squared_error(test_output, predictions)
    mae = mean_absolute_error(test_output, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_output, predictions)

    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("R^2 Score:", r2)

    return mse,mae,rmse,r2
