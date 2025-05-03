# model_results_comparison.py

import os
import pandas as pd
import matplotlib.pyplot as plt

def load_results(results_dir='results'):
    """
    Reads all *_report.csv files in the results_dir and extracts accuracy, macro_precision, macro_recall, macro_f1.
    Returns a DataFrame with one row per model.
    """
    records = []
    for fname in os.listdir(results_dir):
        if fname.endswith('_report.csv'):
            model_name = fname.replace('_report.csv', '').replace('_', ' ')
            path = os.path.join(results_dir, fname)
            report = pd.read_csv(path, index_col=0)
            macro = report.loc['macro avg']
            accuracy = report.loc['accuracy', 'precision']  # accuracy is stored in the 'precision' column
            records.append({
                'model': model_name,
                'accuracy': accuracy,
                'macro_precision': macro['precision'],
                'macro_recall': macro['recall'],
                'macro_f1': macro['f1-score'],
            })
    df = pd.DataFrame(records).set_index('model')
    return df

def plot_macro_f1(df):
    """
    Plots a bar chart of macro-F1 scores for each model.
    """
    df_sorted = df.sort_values('macro_f1', ascending=False)
    plt.figure(figsize=(6,3))
    plt.bar(df_sorted.index, df_sorted['macro_f1'])
    plt.ylabel('Macro‑F1')
    plt.title('Model performance comparison')
    plt.xticks(rotation=15)
    plt.ylim(0, 1.0)
    plt.show()