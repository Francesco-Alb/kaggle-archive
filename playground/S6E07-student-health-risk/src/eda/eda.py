from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

from src.utils.helpers import print_with_sep


@dataclass
class EDA:
    datasets: dict
    target: str
    task_type: str = 'classification'
    add_extra_data: bool = False

    @property
    def focus_numeric_columns(self):
        return self.datasets['train'].select_dtypes(np.number).columns.difference([self.target])

    def numeric_univariate_plots(self):
        focus_cols = self.focus_numeric_columns

        combined_data = pd.concat([
            df.assign(Dataset=name) for name, df in self.datasets.items()
        ])

        for col in focus_cols:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            annot_kws = {'xy': (0.03, 0.75), 'xycoords': 'axes fraction', 'fontsize': 10}

            sns.boxplot(data=combined_data, x=col, y="Dataset", palette="mako", ax=axes[0], boxprops={})
            axes[0].set_xlabel(col)
            axes[0].set_title(f"Box Plot of {col}")

            if combined_data[col].nunique() > 15:
                sns.histplot(data=combined_data, x=col, hue='Dataset', palette="mako", bins=50,
                             stat='density', common_norm=False, multiple='dodge', kde=False, ax=axes[1])
                axes[1].set_xlabel(col)
                axes[1].set_ylabel("Frequency")
                axes[1].set_title(f"Histogram of {col} [{combined_data['Dataset'].unique()}]")
                axes[1].annotate(f"Skewness (TRAIN): {self.datasets['train'][col].skew():.2f}\nKurtosis (TRAIN): {self.datasets['train'][col].kurt():.2f}",
                                 xy=annot_kws['xy'], xycoords=annot_kws['xycoords'], fontsize=annot_kws['fontsize'])
            else:
                sns.countplot(data=combined_data, x=col, hue='Dataset', palette="mako", ax=axes[1])
                axes[1].set_xlabel(col)
                axes[1].set_ylabel("Count")
                axes[1].set_title(f"Countplot of {col} [{combined_data['Dataset'].unique()}]")
                axes[1].annotate(f"Skewness (TRAIN): {self.datasets['train'][col].skew():.2f}\nKurtosis (TRAIN): {self.datasets['train'][col].kurt():.2f}",
                                 xy=annot_kws['xy'], xycoords=annot_kws['xycoords'], fontsize=annot_kws['fontsize'])
            plt.tight_layout()
            plt.show()

    def categorical_plot(self):
        cat_cols = self.datasets["train"].select_dtypes(exclude=np.number).columns.difference([self.target])
        num_plots = len(cat_cols)

        if num_plots == 0:
            print("No categorical features to plot.")
            return
        elif num_plots == 1:
            ncols = 1
            nrows = 1
        else:
            ncols = 2
            nrows = (num_plots + 1) // 2

        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
        ax = axes.flatten() if num_plots > 1 else [axes]

        for i, col in enumerate(cat_cols):
            sns.countplot(data=self.datasets["train"], y=col, order=self.datasets["train"][col].value_counts().index, palette="mako", ax=ax[i])
            ax[i].set_title(f"Distribution of {col}")

        for j in range(num_plots, len(ax)):
            fig.delaxes(ax[j])

        plt.tight_layout()
        plt.show()

    def correlation_plot(self):
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            data=self.datasets["train"].select_dtypes(include=np.number).corr(),
            annot=True,
            cmap="mako",
            linewidth=2,
        )
        plt.tight_layout()
        plt.show()

    def target_plot(self):
        plt.figure(figsize=(12, 5))
        if self.task_type == 'classification':
            sns.countplot(data=self.datasets["train"], y=self.target, order=self.datasets["train"][self.target].value_counts().index, palette='viridis')
        else:
            sns.histplot(data=self.datasets["train"], x=self.target, color='salmon', alpha=0.5)
        plt.title(f'Distribution of {self.target} (Target Variable)')
        plt.xlabel('Count')
        plt.ylabel(f'{self.target}')
        plt.show()

    def print_dataset_overview(self):
        print_with_sep("Shapes")
        for name, df in self.datasets.items():
            print(f"{name} shape: {df.shape}")

        print_with_sep("Duplicates")
        for name, df in self.datasets.items():
            print(f"{name} duplicates: {df.duplicated().sum()}")

        print_with_sep("NaNs")
        for name, df in self.datasets.items():
            print(f"{name} NaNs: {df.isnull().sum().sum()}")

        print_with_sep("Columns not in test")
        print(set(self.datasets['train'].columns).difference(set(self.datasets['test'].columns)))
        if self.add_extra_data:
            print_with_sep("Additional cols in extra data")
            print(set(self.datasets['extra'].columns).difference(set(self.datasets['train'].columns)))

        print_with_sep("Descriptive Statistics")
        for name, df in self.datasets.items():
            print(f"{name} Description:")
            percentage_missing = df.isnull().sum() / df.shape[0]
            percentage_missing.name = '% Missing'
            data_types = df.dtypes
            data_types.name = 'd_type'

            display(
                pd.concat([
                    df.describe(include='all').T,
                    percentage_missing,
                    data_types],
                          axis=1).replace(np.nan, '-').style.background_gradient(cmap='Blues'))
            print("\n")
