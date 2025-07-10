import customtkinter as ctk
from tkinter import messagebox
import functions as f
import alg_functions as alg


class MLApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Aplicație ML - ProgSport")
        self.geometry("1100x850")
        self.data = None
        self.split_var = ctk.StringVar()
        self.input_vars = {}
        self.output_vars = {}

        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.create_widgets()

    def create_widgets(self):
        file_frame = ctk.CTkFrame(self)
        file_frame.pack(padx=20, pady=10, fill="x")
        ctk.CTkLabel(file_frame, text="Înărcare fișier CSV", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=5)
        ctk.CTkButton(file_frame, text="Încarcă fișier", command=self.load_file).pack(pady=5)

        columns_frame = ctk.CTkFrame(self)
        columns_frame.pack(padx=20, pady=10, fill="both", expand=True)

        self.input_checkboxes_frame = ctk.CTkScrollableFrame(columns_frame, label_text="Coloane de Input")
        self.input_checkboxes_frame.pack(side="left", padx=10, pady=10, fill="both", expand=True)

        self.output_checkboxes_frame = ctk.CTkScrollableFrame(columns_frame, label_text="Coloane de Output")
        self.output_checkboxes_frame.pack(side="right", padx=10, pady=10, fill="both", expand=True)

        settings_frame = ctk.CTkFrame(self)
        settings_frame.pack(padx=20, pady=10, fill="x")

        self.algo_var = ctk.StringVar()
        self.tune_var = ctk.StringVar()
        self.norm_var = ctk.StringVar()
        self.split_var = ctk.StringVar()

        ctk.CTkLabel(settings_frame, text="Algoritm:").grid(row=0, column=0, padx=5, pady=5)
        self.combo_algo = ctk.CTkComboBox(settings_frame, variable=self.algo_var, values=[
            'Regresie liniara',
            'Regresie liniara (from zero)',
            'Regresie polinomiala',
            'Regresie Poisson',
            'SVR',
            'Random Forest',
            'Retea neuronala(from zero)',
            'Gradient Boosting',
            'Arbore de decizie'
        ])
        self.combo_algo.grid(row=0, column=1, padx=5, pady=5)

        ctk.CTkLabel(settings_frame, text="Tuning:").grid(row=1, column=0, padx=5, pady=5)
        self.combo_tune = ctk.CTkComboBox(settings_frame, variable=self.tune_var,
                                          values=['Niciuna', 'Grid Search', 'Bayesian Optimization'])
        self.combo_tune.grid(row=1, column=1, padx=5, pady=5)

        ctk.CTkLabel(settings_frame, text="Normalizare:").grid(row=0, column=2, padx=5, pady=5)
        self.combo_norm = ctk.CTkComboBox(settings_frame, variable=self.norm_var,
                                          values=['Niciuna', 'MinMax', 'Standard'])
        self.combo_norm.grid(row=0, column=3, padx=5, pady=5)

        ctk.CTkLabel(settings_frame, text="Split Train-Test:").grid(row=0, column=4, padx=5, pady=5)
        self.combo_split = ctk.CTkComboBox(settings_frame, variable=self.split_var,
                                           values=['90-10', '80-20', '70-30', '60-40', '50-50', '40-60', '35-65'])
        self.combo_split.set("80-20")
        self.combo_split.grid(row=0, column=5, padx=5, pady=5)

        ctk.CTkButton(self, text="Start Train & Test", command=self.run_algorithm).pack(pady=10)

        results_frame = ctk.CTkFrame(self)
        results_frame.pack(padx=20, pady=10, fill="both", expand=True)

        ctk.CTkLabel(results_frame, text="Rezultate", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        self.results_text = ctk.CTkTextbox(results_frame, height=200)
        self.results_text.pack(fill="both", expand=True, padx=5, pady=5)

    def load_file(self):
        self.data = f.select_file()
        if self.data is not None:
            self.data.dropna(inplace=True)
            self.update_columns_list()

    def update_columns_list(self):
        for widget in self.input_checkboxes_frame.winfo_children():
            widget.destroy()
        for widget in self.output_checkboxes_frame.winfo_children():
            widget.destroy()

        self.input_vars = {}
        self.output_vars = {}

        for col in self.data.columns:
            input_var = ctk.BooleanVar()
            output_var = ctk.BooleanVar()
            ctk.CTkCheckBox(self.input_checkboxes_frame, text=col, variable=input_var).pack(anchor="w")
            ctk.CTkCheckBox(self.output_checkboxes_frame, text=col, variable=output_var).pack(anchor="w")
            self.input_vars[col] = input_var
            self.output_vars[col] = output_var

    def get_selected_columns(self, vars_dict):
        return [col for col, var in vars_dict.items() if var.get()]

    def run_algorithm(self):
        if self.data is None:
            messagebox.showerror("Eroare", "Încarcă un fișier mai întâi!")
            return

        input_cols = self.get_selected_columns(self.input_vars)
        output_cols = self.get_selected_columns(self.output_vars)
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
                    metrics, params = alg.optimized_linear_regression_grid_search(train_in, train_out, test_in, test_out)
                else:
                    metrics = alg.linear_regression(train_in, train_out, test_in, test_out)

            elif algorithm == 'Regresie liniara (from zero)':
                if tuning_method == 'Grid Search':
                    metrics, params = alg.optimized_linear_regression_from_zero_grid_search(train_in, train_out, test_in, test_out)
                else:
                    metrics = alg.linear_regression_from_zero(train_in, train_out, test_in, test_out)

            elif algorithm == 'Regresie polinomiala':
                if tuning_method == 'Grid Search':
                    metrics, params = alg.optimized_polynomial_regression_grid_search(train_in, train_out, test_in, test_out)
                elif tuning_method == 'Bayesian Optimization':
                    metrics, params = alg.optimized_polynomial_regression_bayesian(train_in, train_out, test_in, test_out)
                else:
                    metrics = alg.polynomial_regression(train_in, train_out, test_in, test_out, degree=2)

            elif algorithm == 'Regresie Poisson':
                if tuning_method == 'Grid Search':
                    metrics, params = alg.optimized_poisson_regression_grid_search(train_in, train_out, test_in, test_out)
                elif tuning_method == 'Bayesian Optimization':
                    metrics, params = alg.optimized_poisson_regression_bayesian(train_in, train_out, test_in, test_out)
                else:
                    metrics = alg.poisson_regression(train_in, train_out, test_in, test_out, alpha=0.01)

            elif algorithm == 'SVR':
                if tuning_method == 'Grid Search':
                    metrics = alg.optimized_svr_grid_search_saved(train_in, train_out, test_in, test_out)
                elif tuning_method == 'Bayesian Optimization':
                    metrics, params = alg.optimized_svr_bayesian(train_in, train_out, test_in, test_out)
                else:
                    metrics = alg.svr(train_in, train_out, test_in, test_out)

            elif algorithm == 'Random Forest':
                if tuning_method == 'Grid Search':
                    metrics = alg.optimized_random_forest_grid_search_saved(train_in, train_out, test_in, test_out)
                elif tuning_method == 'Bayesian Optimization':
                    metrics = alg.optimized_random_forest_bayesian_saved(train_in, train_out, test_in, test_out)
                else:
                    metrics = alg.random_forest_features_importances(train_in, train_out, test_in, test_out, feature_names)

            elif algorithm == 'Retea neuronala(from zero)':
                if tuning_method == 'Grid Search':
                    metrics = alg.optimized_neural_network_from_zero_grid_search_saved(train_in, train_out, test_in, test_out)
                elif tuning_method == 'Bayesian Optimization':
                    metrics = alg.optimized_neural_network_from_zero_bayesian_saved(train_in, train_out, test_in, test_out)
                else:
                    metrics = alg.neural_network_from_zero(train_in, train_out, test_in, test_out)

            elif algorithm == 'Gradient Boosting':
                if tuning_method == 'Grid Search':
                    metrics = alg.optimized_gradient_boosting_grid_search_saved(train_in, train_out, test_in, test_out)
                elif tuning_method == 'Bayesian Optimization':
                    metrics = alg.optimized_gradient_boosting_bayesian_saved(train_in, train_out, test_in, test_out)
                else:
                    metrics = alg.gradient_boosting(train_in, train_out, test_in, test_out)

            elif algorithm == 'Arbore de decizie':
                if tuning_method == 'Grid Search':
                    metrics, params = alg.optimized_decision_tree_grid_search(train_in, train_out, test_in, test_out)
                elif tuning_method == 'Bayesian Optimization':
                    metrics, params = alg.optimized_decision_tree_bayesian(train_in, train_out, test_in, test_out)
                else:
                    metrics = alg.decision_tree_features_importance(train_in, train_out, test_in, test_out, max_depth=5, feature_names=feature_names)
            else:
                messagebox.showerror("Eroare", "Algoritm necunoscut!")
                return

            self.results_text.delete("1.0", ctk.END)
            self.results_text.insert(ctk.END, "Metrici de evaluare:\n\n")
            self.results_text.insert(ctk.END, f"MSE: {metrics[0]}\n")
            self.results_text.insert(ctk.END, f"MAE: {metrics[1]}\n")
            self.results_text.insert(ctk.END, f"RMSE: {metrics[2]}\n")
            self.results_text.insert(ctk.END, f"R²: {metrics[3]}\n")
        except Exception as e:
            messagebox.showerror("Eroare", f"A apărut o eroare: {str(e)}")


if __name__ == "__main__":
    app = MLApp()
    app.mainloop()


