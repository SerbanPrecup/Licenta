import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd


class CSVColumnSelector:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Column Selector")
        self.file_paths = []
        self.common_columns = []

        self.label = tk.Label(root, text="Select CSV files:")
        self.label.pack(pady=10)

        self.browse_btn = tk.Button(root, text="Search", command=self.browse_files)
        self.browse_btn.pack(pady=5)

    def browse_files(self):
        self.file_paths = filedialog.askopenfilenames(
            title="Select CSV files:",
            filetypes=(("CSV Files", "*.csv"), ("All files", "*.*"))
        )

        if self.file_paths:
            try:
                column_sets = []
                for file in self.file_paths:
                    df = pd.read_csv(file, nrows=0)
                    column_sets.append(set(df.columns))

                self.common_columns = list(set.intersection(*column_sets))

                if not self.common_columns:
                    messagebox.showerror("Error", "There are no common columns between files!")
                    return

                self.show_column_selector()

            except Exception as e:
                messagebox.showerror("Error", f"Reading files error:\n{e}")

    def show_column_selector(self):
        self.top = tk.Toplevel(self.root)
        self.top.title("Select columns")

        frame = tk.Frame(self.top)
        frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.listbox = tk.Listbox(
            frame,
            selectmode=tk.MULTIPLE,
            activestyle="none",
            height=15
        )
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.listbox.yview)

        self.common_columns.sort()
        for col in self.common_columns:
            self.listbox.insert(tk.END, col)

        btn_frame = tk.Frame(self.top)
        btn_frame.pack(pady=10)

        tk.Button(
            btn_frame,
            text="Processing",
            command=self.process_columns
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame,
            text="Cancel",
            command=self.top.destroy
        ).pack(side=tk.LEFT, padx=5)

    def process_columns(self):
        selected = self.listbox.curselection()
        if not selected:
            messagebox.showwarning("Advertisement", "You have not selected any columns!")
            return

        selected_cols = [self.common_columns[i] for i in selected]

        try:
            dfs = []
            for file in self.file_paths:
                df = pd.read_csv(file, usecols=selected_cols)
                dfs.append(df)

            final_df = pd.concat(dfs, ignore_index=True)

            save_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=(("CSV Files", "*.csv"), ("All files", "*.*")),
                title="Save the concatenated file"
            )

            if save_path:
                final_df.to_csv(save_path, index=False)
                messagebox.showinfo("Success", "The concatenated file was saved successfully!")
                self.top.destroy()

        except Exception as e:
            messagebox.showerror("Error", f"Processing Error:\n{e}")



root = tk.Tk()
app = CSVColumnSelector(root)
root.mainloop()