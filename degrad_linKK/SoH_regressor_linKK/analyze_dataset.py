import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sanitizing_utils import data_sanitize

# Path to the dataset
DATA_PATH_GEN = '/home/skywalk3r/Scrivania/data_augmentation_alessio/generated_data_sanitized.csv'
DATA_PATH = '/home/skywalk3r/Scrivania/SoH_GPR_customKernel/dataset/dataset_all.csv'
DATA_PATH_AUG = '/home/skywalk3r/Scrivania/data_augmentation_alessio/augmented_data.csv'
DATA_PATH_TBDPM = '/home/skywalk3r/Scrivania/data_augmentation_alessio/tab_ddpm_easyrun/TabDDPM_easyRun/DATI_SINTETICI_TABDDPM.csv'
DATA_PATH_SPLIT = '/home/skywalk3r/Scrivania/data_augmentation_alessio/generated_data_split.csv'

def load_data(path):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"Data shape: {df.shape}")
    print("Columns:", df.columns.tolist()[:10], "...")
    return df

def basic_stats(df):
    print("\nBasic Statistics:")
    print(df[['SoH', 'Temp', 'SoC']].describe())

# --- FUNZIONE CORRETTA ---
def plot_nyquist(df, ax, title, samples_per_temp=127):
    """
    Plots Nyquist diagrams on the given 3D axis 'ax'.
    """
    unique_temps = sorted(df['Temp'].unique())
    
    # Identify impedance columns
    cols = df.columns
    re_cols = [c for c in cols if c.endswith('-Re')]
    im_cols = [c for c in cols if c.endswith('-Im')]
    
    for temp in unique_temps:
        subset = df[df['Temp'] == temp]
        
        if len(subset) > samples_per_temp:
            sample = subset.sample(n=samples_per_temp, random_state=42)
        else:
            sample = subset
            
        for idx, row in sample.iterrows():
            re_vals = row[re_cols].values
            im_vals = row[im_cols].values
            
            # Z values (Temperature)
            temp_vals = np.full(re_vals.shape, temp)
            
            # Color by SoH
            soh = row['SoH']
            color = plt.cm.coolwarm((soh - 80) / 20)
            
            # Plot su 'ax' passato come argomento
            ax.plot(temp_vals, re_vals, -im_vals, color=color, alpha=0.6, marker='o', markersize=1)

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Z' (Real)")
    ax.set_zlabel("-Z'' (Imaginary)")
    ax.set_title(title)
   

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: File not found at {DATA_PATH}")
        return

    df = load_data(DATA_PATH)
    df_gen = load_data(DATA_PATH_GEN)
    #df_augmented = load_data(DATA_PATH_AUG)
    df_split = load_data(DATA_PATH_SPLIT)
    df_split = data_sanitize(df_split)
    df_tbddpm = load_data(DATA_PATH_TBDPM)

    basic_stats(df)
    basic_stats(df_gen)
    #basic_stats(df_augmented)
    basic_stats(df_split)
    basic_stats(df_tbddpm)
    
    print("Generating side-by-side comparison plots...")

    # Creiamo la figura e gli assi (1 riga, 2 colonne) specificando la proiezione 3D
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), subplot_kw={'projection': '3d'})

    # Disegna i grafici
    plot_nyquist(df, axes[0], "REAL Data")
    #plot_nyquist(df_augmented, axes[1], "GENERATED Data")
    plot_nyquist(df_split, axes[1], "VAE GEN Data")
    plot_nyquist(df_tbddpm, axes[2], "TBDDPM Data")

    # Aggiungi una colorbar comune per tutta la figura
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=80, vmax=100))
    sm.set_array([])
    
    # ax=axes.ravel().tolist() serve a centrare la colorbar rispetto a entrambi i grafici
    fig.colorbar(sm, ax=axes.ravel().tolist(), label='SoH (State of Health) %', shrink=0.5, pad=0.05)

    plt.show()

if __name__ == "__main__":
    main()