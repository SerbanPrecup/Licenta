import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np

import functions as f
import alg_functions as alg
import pandas as pd
import os

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Aplicație ML - ProgSport")
        self.geometry("1000x800")
        self.data = None
        self.filename = None
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
                                           'Retea neuronala(Keras)',
                                           'Retea neuronala(from zero)',
                                           'Gradient Boosting',
                                           'Arbore de decizie'
                                       ], state='readonly')
        self.combo_algo.grid(row=0, column=1, padx=5, pady=5)

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

        self.btn_run = ttk.Button(self, text="Antrenează modelul", command=self.run_algorithm)
        self.btn_run.pack(pady=10)

        results_frame = ttk.LabelFrame(self, text="Rezultate")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.results_text = tk.Text(results_frame, wrap=tk.WORD, height=10)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def load_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv")],
            title="Alege fișier CSV"
        )
        if not path:
            return
        self.filename = os.path.basename(path)
        self.data = pd.read_csv(path)
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
        const_cols = [c for c in input_cols if self.data[c].nunique() <= 1]
        if const_cols:
            messagebox.showwarning(
                "Atenție",
                f"Atenție, următoarele coloane au toată valoarea fixă și vor fi eliminate: {const_cols}"
            )
            input_cols = [c for c in input_cols if c not in const_cols]
        algo = self.algo_var.get()

        if not input_cols or not output_cols:
            messagebox.showerror("Eroare", "Selectează coloane pentru input și output!")
            return
        if not algo:
            messagebox.showerror("Eroare", "Alege un algoritm!")
            return

        available_tuning = {
            'Regresie liniara': ['Niciuna', 'Grid Search'],
            'Regresie liniara (from zero)': ['Niciuna', 'Grid Search','Bayesian Optimization'],
            'Regresie polinomiala': ['Niciuna', 'Grid Search','Bayesian Optimization'],
            'Regresie Poisson': ['Niciuna', 'Grid Search','Bayesian Optimization'],
            'SVR': ['Niciuna', 'Grid Search','Bayesian Optimization'],
            'Random Forest': ['Niciuna', 'Grid Search','Bayesian Optimization'],
            'Retea neuronala(Keras)': ['Niciuna', 'Grid Search','Bayesian Optimization'],
            'Retea neuronala(from zero)': ['Niciuna', 'Grid Search','Bayesian Optimization'],
            'Gradient Boosting': ['Niciuna', 'Grid Search','Bayesian Optimization'],
            'Arbore de decizie': ['Niciuna', 'Grid Search','Bayesian Optimization']
        }
        tune_methods = available_tuning.get(algo, ['Niciuna'])

        try:
            splits = ['90-10', '80-20', '70-30', '60-40', '50-50', '40-60', '35-65']
            all_rows = []

            for split_ratio in splits:
                pct = int(split_ratio.split('-')[0])
                tr_i, tr_o, te_i, te_o = f.shuffle_division(
                    self.data, pct, input_cols, output_cols
                )
                mask = ~np.isnan(tr_i).any(axis=1)
                tr_i, tr_o = tr_i[mask], tr_o[mask]
                mask = ~np.isnan(te_i).any(axis=1)
                te_i, te_o = te_i[mask], te_o[mask]

                if algo == 'Regresie Poisson':
                    datasets = {
                        'MIN MAX': f.normalize_min_max(tr_i, tr_o, te_i, te_o),
                        'STANDARD': f.normalize_min_max(tr_i, tr_o, te_i, te_o)
                    }
                else:
                    datasets = {
                        'MIN MAX': f.normalize_min_max(tr_i, tr_o, te_i, te_o),
                        'STANDARD': f.normalize_standard(tr_i, tr_o, te_i, te_o)
                    }

                for tune in tune_methods:
                    metrics_results = {}
                    for norm_name, (ti, to, vi, vo) in datasets.items():
                        if algo == 'Regresie liniara':
                            if tune == 'Grid Search':
                                m, _ = alg.optimized_linear_regression_grid_search(ti, to, vi, vo)
                            else:
                                m = alg.linear_regression(ti, to, vi, vo)

                        elif algo == 'Regresie liniara (from zero)':
                            if tune == 'Grid Search':
                                m, _ = alg.optimized_linear_regression_from_zero_grid_search(ti, to, vi, vo)
                            elif tune == 'Bayesian Optimization':
                                m, _ = alg.optimized_linear_regression_from_zero_bayesian(ti, to, vi, vo)
                            else:
                                m = alg.linear_regression_from_zero(ti, to, vi, vo)

                        elif algo == 'Regresie polinomiala':
                            if tune == 'Grid Search':
                                m, _ = alg.optimized_polynomial_regression_grid_search(ti, to, vi, vo)
                            elif tune == 'Bayesian Optimization':
                                m = alg.optimized_polynomial_regression_bayesian_saved(ti, to, vi, vo)
                            else:
                                m = alg.polynomial_regression(ti, to, vi, vo, degree=2)

                        elif algo == 'Regresie Poisson':
                            if tune == 'Grid Search':
                                m, _ = alg.optimized_poisson_regression_grid_search(ti, to, vi, vo)
                            elif tune == 'Bayesian Optimization':
                                m, _ = alg.optimized_poisson_regression_bayesian(ti, to, vi, vo)
                            else:
                                m = alg.poisson_regression(ti, to, vi, vo, alpha=0.01)

                        elif algo == 'SVR':
                            if tune == 'Grid Search':
                                m = alg.optimized_svr_grid_search_saved(ti, to, vi, vo)
                            elif tune == 'Bayesian Optimization':
                                m = alg.optimized_svr_bayesian_saved(ti, to, vi, vo)
                            else:
                                m = alg.svr(ti, to, vi, vo)

                        elif algo == 'Random Forest':
                            if tune == 'Grid Search':
                                m = alg.optimized_random_forest_grid_search_saved(ti, to, vi, vo)
                            elif tune == 'Bayesian Optimization':
                                m = alg.optimized_random_forest_bayesian_saved(ti, to, vi, vo)
                            else:
                                m = alg.random_forest(ti, to, vi, vo)

                        elif algo == 'Retea neuronala(Keras)':
                            if tune == 'Grid Search':
                                m = alg.optimized_neural_network_grid_search(ti, to, vi, vo)
                            elif tune == 'Bayesian Optimization':
                                m, _ = alg.optimized_neural_network_bayesian(ti, to, vi, vo)
                            else:
                                m = alg.neural_network(ti, to, vi, vo)

                        elif algo == 'Retea neuronala(from zero)':
                            if tune == 'Grid Search':
                                m = alg.optimized_neural_network_from_zero_grid_search_saved(ti, to, vi, vo)
                            elif tune == 'Bayesian Optimization':
                                m = alg.optimized_neural_network_from_zero_bayesian_saved(ti, to, vi, vo)
                            else:
                                m = alg.neural_network_from_zero(ti, to, vi, vo)

                        elif algo == 'Gradient Boosting':
                            if tune == 'Grid Search':
                                m = alg.optimized_gradient_boosting_grid_search_saved(ti, to, vi, vo)
                            elif tune == 'Bayesian Optimization':
                                m = alg.optimized_gradient_boosting_bayesian_saved(ti, to, vi, vo)
                            else:
                                m = alg.gradient_boosting(ti, to, vi, vo)

                        elif algo == 'Arbore de decizie':
                            if tune == 'Grid Search':
                                m, _ = alg.optimized_decision_tree_grid_search(ti, to, vi, vo)
                            elif tune == 'Bayesian Optimization':
                                m, _ = alg.optimized_decision_tree_bayesian(ti, to, vi, vo)
                            else:
                                m = alg.decision_tree(ti, to, vi, vo,max_depth=5,)
                        else:
                            messagebox.showerror("Eroare", "Algoritm necunoscut!")
                            return

                        metrics_results[norm_name] = m

                    all_rows.append({
                        'nume_fisier':       self.filename,
                        'algoritm':          algo,
                        'input':             ','.join(input_cols),
                        'procente':          pct,
                        'MSE - MIN MAX':     metrics_results['MIN MAX'][0],
                        'MAE - MIN MAX':     metrics_results['MIN MAX'][1],
                        'RMSE - MIN MAX':    metrics_results['MIN MAX'][2],
                        'R^2 - MIN MAX':     metrics_results['MIN MAX'][3],
                        'MSE - STANDARD':    metrics_results['STANDARD'][0],
                        'MAE - STANDARD':    metrics_results['STANDARD'][1],
                        'RMSE - STANDARD':   metrics_results['STANDARD'][2],
                        'R^2 - STANDARD':    metrics_results['STANDARD'][3],
                        'metoda de tunning': tune
                    })

            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Rezultate (split × tuning × normalizare):\n\n")
            for r in all_rows:
                self.results_text.insert(
                    tk.END,
                    f"{r['procente']}% | {r['metoda de tunning']} | "
                    f"MSE_MINMAX={r['MSE - MIN MAX']}, MSE_STD={r['MSE - STANDARD']}\n"
                )

            df = pd.DataFrame(all_rows, columns=[
                'nume_fisier','algoritm','input','procente',
                'MSE - MIN MAX','MAE - MIN MAX','RMSE - MIN MAX','R^2 - MIN MAX',
                'MSE - STANDARD','MAE - STANDARD','RMSE - STANDARD','R^2 - STANDARD',
                'metoda de tunning'
            ])
            fname = "statistica-gui.xlsx"
            try:
                import openpyxl
                engine = 'openpyxl'
            except ImportError:
                engine = 'xlsxwriter'

            if not os.path.exists(fname):
                df.to_excel(fname, index=False, engine=engine)
            else:
                with pd.ExcelWriter(fname, engine=engine, mode='a', if_sheet_exists='overlay') as writer:
                    startrow = writer.sheets['Sheet1'].max_row
                    df.to_excel(writer, index=False, header=False, startrow=startrow)

            messagebox.showinfo("Gata",
                f"{len(all_rows)} rânduri salvate în:\n{os.path.abspath(fname)}")

        except Exception as e:
            messagebox.showerror("Eroare", f"A apărut o eroare: {e}")





# if __name__ == "__main__":
app = Application()
app.mainloop()
