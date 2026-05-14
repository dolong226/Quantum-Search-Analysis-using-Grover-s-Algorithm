from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src.grover import *
from src.noise_model import *

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)



# Cấu hình đồ thị
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "monospace",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# Color
COLOR = {
    "quantum":    "#2196F3", 
    "theory":     "#4CAF50",  
    "simulated":  "#FF9800",   
    "ideal":      "#9C27B0",   
    "noise":      "#795548",  
    "threshold":  "#607D8B",   # ngưỡng
}

# Lưu file

def _save_figure(fig: plt.Figure, filename: str) -> Path:
    filepath = RESULTS_DIR / f"{filename}.png"
    fig.savefig(filepath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {filepath}")
    return filepath

def _save_dataframe(df: pd.DataFrame, filename: str) -> Path:
    filepath = RESULTS_DIR / f"{filename}.csv"
    df.to_csv(filepath, index=False, float_format="%.6f")
    print(f"Saved: {filepath}")
    return filepath


