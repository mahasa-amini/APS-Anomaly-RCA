# src/data_loading.py
import pandas as pd
import numpy as np
from .config import TRAIN_FILE

def load_raw_aps() -> pd.DataFrame:
    """
    Load the raw APS training dataset
    Automatically detects delimiter.
    """
    df = pd.read_csv(TRAIN_FILE, sep=None, engine="python")
    df = df.replace("na", np.nan)
    return df

