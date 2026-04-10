import numpy as np
import pandas as pd
import re

VALID_TEMPS = np.array([15, 25, 35])
VALID_SOCS = np.array([5, 20, 50, 70, 95])
VALID_CELLS = np.array([2,3,4,5,6,12,13,14,15,17,18,19,20,21,22,23,24,25,26,28,29,30,31,32])
VALID_SOHS = np.array([80,85,90,95,100])

def data_sanitize(df):
    """
    Sanitizes ['Cell_Name', 'Temp', 'SoC', 'SoH'] columns by snapping 
    continuous values to the nearest valid discrete option.
    """
    df_clean = df.copy()
    
    if 'Cell_Name' in df_clean.columns:
        df_clean['Cell_Name'] = df_clean['Cell_Name'].apply(lambda x: VALID_CELLS[(np.abs(VALID_CELLS-x)).argmin()])
    elif 'Cell_name' in df_clean.columns: # Gestione alternativa
        df_clean['Cell_name'] = df_clean['Cell_name'].apply(lambda x: VALID_CELLS[(np.abs(VALID_CELLS-x)).argmin()])

    df_clean['Temp'] = df_clean['Temp'].apply(lambda x: VALID_TEMPS[(np.abs(VALID_TEMPS-x)).argmin()])
    df_clean['SoC'] = df_clean['SoC'].apply(lambda x: VALID_SOCS[(np.abs(VALID_SOCS-x)).argmin()])
    df_clean['SoH'] = df_clean['SoH'].apply(lambda x: VALID_SOHS[(np.abs(VALID_SOHS-x)).argmin()])
    
    return df_clean



def get_sorted_impedance_cols(columns):
    # Trova colonne che finiscono con -Re o -Im
    re_cols = [c for c in columns if c.endswith('-Re')]
    im_cols = [c for c in columns if c.endswith('-Im')]
    
    # Estrai la frequenza numerica dal nome (es. "10000-Re" -> 10000.0)
    def extract_freq(col_name):
        match = re.match(r"([0-9\.]+)-", col_name)
        return float(match.group(1)) if match else 0

    # Ordina decrescente (da alta a bassa frequenza per Nyquist)
    re_cols.sort(key=extract_freq, reverse=True)
    im_cols.sort(key=extract_freq, reverse=True)
    
    return re_cols, im_cols
