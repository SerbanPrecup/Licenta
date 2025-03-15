import sys

import numpy as np
import pandas as pd
from tkinter.filedialog import askopenfilename

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

def normalize_min_max(input_data, output_data):
    min_max_scaler = MinMaxScaler()
    input_data = min_max_scaler.fit_transform(input_data)
    output_data = min_max_scaler.fit_transform(output_data)
    return input_data, output_data

def normalize_standard(input_data, output_data):
    standard_scaler = StandardScaler()
    input_data = standard_scaler.fit_transform(input_data)
    output_data = standard_scaler.fit_transform(output_data)
    return input_data, output_data

def shuffle_division(data,train_percent,input_columns,output_columns):
    data = data.sample(frame=1, random_state=42).reset_index(drop=True)
    train_size = int(train_percent/100 * len(data))
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    #extraction
    train_input = train_data[input_columns].values.astype(np.float32)
    train_output = train_data[output_columns].values.astype(np.float32)
    test_input = test_data[input_columns].values.astype(np.float32)
    test_output = test_data[output_columns].values.astype(np.float32)

    return train_input, train_output, test_input, test_output


