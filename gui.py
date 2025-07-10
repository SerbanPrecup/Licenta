import tkinter as tk
from tkinter import ttk, messagebox
import functions as f
import alg_functions as alg
import pandas as pd


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Aplicație ML - ProgSport")
        self.geometry("1000x800")
        self.data = None
        self.split_var = tk.StringVar()
        self.create_widgets()

    def create_widgets(self):
        file_frame = ttk.LabelFrame(self, text="Încărcare fișier CSV")
        file_frame.pack(fill=tk.X, padx=10, pady=5)

        self.btn_load = ttk.Button(file_frame, text="Încarcă fișier", command=self.load_file)
        self.btn_load.pack(pady=5)

        columns_frame = ttk.LabelFrame(self, text="Selectare coloane")
        columns_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.listbox_input = tk.Listbox(columns_frame, selectmode=tk.MULTIPLE, exportselection=False, height=10)
        self.listbox_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        ttk.Label(columns_frame, text="Input Columns").pack(side=tk.LEFT)

        self.listbox_output = tk.Listbox(columns_frame, selectmode=tk.MULTIPLE, exportselection=False, height=10)
        self.listbox_output.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        ttk.Label(columns_frame, text="Output Columns").pack(side=tk.RIGHT)

        settings_frame = ttk.LabelFrame(self, text="Configurare model")
        settings_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(settings_frame, text="Alege algoritmul:").grid(row=0, column=0, padx=5, pady=5)
        self.algo_var = tk.StringVar()
        self.combo_algo = ttk.Combobox(settings_frame, textvariable=self.algo_var,
                                       values=[
                                           'Regresie liniara',
                                           'Regresie liniara (from zero)',
                                           'Regresie polinomiala',
                                           'Regresie Poisson',
                                           'SVR',
                                           'Random Forest',
                                           'Retea neuronala(from zero)',
                                           'Gradient Boosting',
                                           'Arbore de decizie'
                                       ], state='readonly')
        self.combo_algo.grid(row=0, column=1, padx=5, pady=5)

        # settings_frame = ttk.LabelFrame(self, text="Configurare model")
        # settings_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(settings_frame, text="Metoda de tuning:").grid(row=1, column=0, padx=5, pady=5)
        self.tune_var = tk.StringVar()
        self.combo_tune = ttk.Combobox(settings_frame, textvariable=self.tune_var,
                                       values=['Niciuna', 'Grid Search', 'Bayesian Optimization'],
                                       state='readonly')
        self.combo_tune.grid(row=1, column=1, padx=5, pady=5)
        self.combo_tune.set('Niciuna')

        ttk.Label(settings_frame, text="Normalizare:").grid(row=0, column=2, padx=5, pady=5)
        self.norm_var = tk.StringVar()
        self.combo_norm = ttk.Combobox(settings_frame, textvariable=self.norm_var,
                                       values=['MinMax', 'Standard', 'Niciuna'], state='readonly')
        self.combo_norm.grid(row=0, column=3, padx=5, pady=5)
        self.combo_norm.set('Niciuna')

        ttk.Label(settings_frame, text="Split Train-Test:").grid(row=0, column=4, padx=5, pady=5)
        self.split_var = tk.StringVar()
        self.combo_split = ttk.Combobox(settings_frame, textvariable=self.split_var,
                                        values=['90-10', '80-20', '70-30', '60-40', '50-50', '40-60', '35-65'],
                                        state='readonly')
        self.combo_split.grid(row=0, column=5, padx=5, pady=5)
        self.combo_split.set('80-20')

        self.btn_run = ttk.Button(self, text="Start Train & Test", command=self.run_algorithm)
        self.btn_run.pack(pady=10)

        results_frame = ttk.LabelFrame(self, text="Rezultate")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.results_text = tk.Text(results_frame, wrap=tk.WORD, height=10)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def load_file(self):
        self.data = f.select_file()
        if self.data is not None:
            self.data.dropna(inplace=True)
            self.update_columns_list()

    def update_columns_list(self):
        self.listbox_input.delete(0, tk.END)
        self.listbox_output.delete(0, tk.END)
        for col in self.data.columns:
            self.listbox_input.insert(tk.END, col)
            self.listbox_output.insert(tk.END, col)

    def get_selected_columns(self, listbox):
        return [listbox.get(i) for i in listbox.curselection()]

    def run_algorithm(self):
        if self.data is None:
            messagebox.showerror("Eroare", "Încarcă un fișier mai întâi!")
            return

        input_cols = self.get_selected_columns(self.listbox_input)
        output_cols = self.get_selected_columns(self.listbox_output)
        feature_names = input_cols

        if not input_cols or not output_cols:
            messagebox.showerror("Eroare", "Selectează coloane pentru input și output!")
            return

        algorithm = self.algo_var.get()
        tuning_method = self.tune_var.get()
        available_tuning = {
            'Regresie liniara': ['Niciuna', 'Grid Search'],
            'Regresie liniara (from zero)': ['Niciuna', 'Grid Search', 'Bayesian Optimization'],
            'Regresie polinomiala': ['Niciuna', 'Grid Search', 'Bayesian Optimization'],
            'Regresie Poisson': ['Niciuna', 'Grid Search', 'Bayesian Optimization'],
            'SVR': ['Niciuna', 'Grid Search', 'Bayesian Optimization'],
            'Random Forest': ['Niciuna', 'Grid Search', 'Bayesian Optimization'],
            'Retea neuronala(from zero)': ['Niciuna', 'Grid Search', 'Bayesian Optimization'],
            'Gradient Boosting': ['Niciuna', 'Grid Search', 'Bayesian Optimization'],
            'Arbore de decizie': ['Niciuna', 'Grid Search', 'Bayesian Optimization']
        }

        if tuning_method not in available_tuning.get(algorithm, ['Niciuna']):
            messagebox.showerror("Eroare", f"Metoda {tuning_method} nu este disponibilă pentru {algorithm}")
            return
        normalization = self.norm_var.get()

        if not algorithm:
            messagebox.showerror("Eroare", "Alege un algoritm!")
            return

        try:
            split_ratio = self.split_var.get()
            train_percent = int(split_ratio.split('-')[0])

            train_input, train_output, test_input, test_output = f.shuffle_division(
                self.data, train_percent, input_cols, output_cols
            )

            if normalization == 'MinMax':
                train_in, train_out, test_in, test_out = f.normalize_min_max(
                    train_input, train_output, test_input, test_output
                )
            elif normalization == 'Standard':
                train_in, train_out, test_in, test_out = f.normalize_standard(
                    train_input, train_output, test_input, test_output
                )
            else:
                train_in, train_out, test_in, test_out = train_input, train_output, test_input, test_output

            params = {}
            if algorithm == 'Regresie liniara':
                if tuning_method == 'Grid Search':
                    metrics, params = alg.optimized_linear_regression_grid_search(train_in, train_out,
                                                                                            test_in, test_out)
                else:
                    metrics = alg.linear_regression(train_in, train_out, test_in, test_out)

            elif algorithm == 'Regresie liniara (from zero)':
                if tuning_method == 'Grid Search':
                    metrics, params = alg.optimized_linear_regression_from_zero_grid_search(train_in, train_out,
                                                                                            test_in,
                                                                                            test_out)
                else:
                    metrics = alg.linear_regression_from_zero(train_in, train_out, test_in, test_out)

            elif algorithm == 'Regresie polinomiala':
                if tuning_method == 'Grid Search':
                    metrics = alg.optimized_polynomial_regression_grid_search(train_in, train_out, test_in, test_out)
                elif tuning_method == 'Bayesian Optimization':
                    metrics, params = alg.optimized_polynomial_regression_bayesian(train_in, train_out, test_in,
                                                                                   test_out)
                else:
                    metrics = alg.polynomial_regression(train_in, train_out, test_in, test_out, degree=2)

            elif algorithm == 'Regresie Poisson':
                if tuning_method == 'Grid Search':
                    metrics = alg.optimized_poisson_regression_grid_search(train_in, train_out, test_in, test_out)
                elif tuning_method == 'Bayesian Optimization':
                    metrics, params = alg.optimized_poisson_regression_bayesian(train_in, train_out, test_in, test_out)
                else:
                    metrics = alg.poisson_regression(train_in, train_out, test_in, test_out, alpha=0.01)

            elif algorithm == 'SVR':
                if tuning_method == 'Grid Search':
                    metrics = alg.optimized_svr_grid_search(train_in, train_out, test_in, test_out)
                elif tuning_method == 'Bayesian Optimization':
                    metrics, params = alg.optimized_svr_bayesian(train_in, train_out, test_in, test_out)
                else:
                    metrics = alg.svr(train_in, train_out, test_in, test_out)

            elif algorithm == 'Random Forest':
                if tuning_method == 'Grid Search':
                    metrics = alg.optimized_random_forest_grid_search(train_in, train_out, test_in, test_out)
                elif tuning_method == 'Bayesian Optimization':
                    metrics, params = alg.optimized_random_forest_bayesian(train_in, train_out, test_in, test_out)
                else:
                    metrics = alg.random_forest_features_importances(train_in, train_out, test_in, test_out,
                                                                     feature_names)

            elif algorithm == 'Retea neuronala(from zero)':
                if tuning_method == 'Grid Search':
                    metrics = alg.optimized_neural_network_from_zero_grid_search(train_in, train_out, test_in, test_out)
                elif tuning_method == 'Bayesian Optimization':
                    metrics, params = alg.optimized_neural_network_from_zero_bayesian(train_in, train_out, test_in,
                                                                                      test_out)
                else:
                    metrics = alg.neural_network_from_zero(train_in, train_out, test_in, test_out)

            elif algorithm == 'Gradient Boosting':
                if tuning_method == 'Grid Search':
                    metrics = alg.optimized_gradient_boosting_grid_search(train_in, train_out, test_in, test_out)
                elif tuning_method == 'Bayesian Optimization':
                    metrics, params = alg.optimized_gradient_boosting_bayesian(train_in, train_out, test_in, test_out)
                else:
                    metrics = alg.gradient_boosting(train_in, train_out, test_in, test_out)

            elif algorithm == 'Arbore de decizie':
                if tuning_method == 'Grid Search':
                    metrics = alg.optimized_decision_tree_grid_search(train_in, train_out, test_in, test_out)
                elif tuning_method == 'Bayesian Optimization':
                    metrics, params = alg.optimized_decision_tree_bayesian(train_in, train_out, test_in, test_out)
                else:
                    metrics = alg.decision_tree_features_importance(train_in, train_out, test_in, test_out, max_depth=5,
                                                                    feature_names=feature_names)
            else:
                messagebox.showerror("Eroare", "Algoritm necunoscut!")
                return

            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Metrici de evaluare:\n\n")
            self.results_text.insert(tk.END, f"MSE: {metrics[0]:}\n")
            self.results_text.insert(tk.END, f"MAE: {metrics[1]:}\n")
            self.results_text.insert(tk.END, f"RMSE: {metrics[2]:}\n")
            self.results_text.insert(tk.END, f"R²: {metrics[3]:}\n")
        except Exception as e:
            messagebox.showerror("Eroare", f"A apărut o eroare: {str(e)}")


# if __name__ == "__main__":
app = Application()
app.mainloop()
