import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from sanitizing_utils import data_sanitize
from KK_test import get_sorted_impedance_cols
from analyze_dataset import plot_nyquist

DATA_PATH = 'dataset_all.csv'
CUTOFF_LOW = 5e-2
CUTOFF_HIGH = 1e3

COEFF_AMP_BROAD = 0.1
COEFF_AMP_NARROW = 0.1

def noise(Z_complex_gt, freqs):
    """
    Introduco il rumore in maniera frequency dependent
    """
    Z_complex_abs_gt = np.abs(Z_complex_gt)
    
    # disturbo a banda larga amp = [1,3,5,10] %
    e_broad_amp = COEFF_AMP_BROAD * Z_complex_abs_gt
    """
    L'oggetto freq dep comp. è interpretabile come un filtro elimina banda
    """
    freq_dep_comp = 1/(1+(2*np.pi*freqs)/(2*np.pi*CUTOFF_LOW)) + 1/(1+(2*np.pi*CUTOFF_HIGH)/(2*np.pi*freqs))
    e_rand_env = e_broad_amp * freq_dep_comp

    """
    Genero un vettore di 61 numeri casuali a media 0 e varianza 1
    """
    n_points = len(freqs)
    e_rand_broad = (np.random.normal(0,1, n_points) + 1j*np.random.normal(0,1, n_points)) * e_rand_env

    # disturbo a banda stretta (amp: 10%)
    e_narrow_amp = COEFF_AMP_NARROW * Z_complex_abs_gt 

    #vett rumore vuoto
    noise_narrow = np.zeros_like(Z_complex_gt, dtype=complex)

    # indici rispetto a cui applico il rumore
    idx_low  = np.where(freqs <= CUTOFF_LOW)[0]
    idx_high = np.where(freqs >= CUTOFF_HIGH)[0]

    """
    Vado a generare 2 numeri casuali per ogni banda
    """\
    #applicazione rumore
    if len(idx_low) >= 2:
        pts = np.random.choice(idx_low, 2, replace=False)
        noise_narrow[pts] = (np.random.normal(0, 1, 2) + 1j*np.random.normal(0, 1, 2)) * e_narrow_amp[pts]

 
    if len(idx_high) >= 2:
        pts = np.random.choice(idx_high, 2, replace=False)
        noise_narrow[pts] = (np.random.normal(0, 1, 2) + 1j*np.random.normal(0, 1, 2)) * e_narrow_amp[pts]

    e_narrow_rand = noise_narrow

    #aggiungo il rumore totale al vettore di ground truth
    Z_complex_noisy = Z_complex_gt + (e_rand_broad + e_narrow_rand)
    return Z_complex_noisy



def scatter_plot(df,df_noisy):

    """
    Scatter plot acquisizioni
    """
    
    cols = df.columns
    re_cols = [c for c in cols if c.endswith('-Re')]
    im_cols = [c for c in cols if c.endswith('-Im')]
    
    # le temp sono [15, 25, 35], ne fisso una per esempio
    n_samples = 1
    temp = 15
    subset = df[df['Temp'] == temp]
    if len(subset) > n_samples:
        subset = subset.sample(n=n_samples, random_state=42)

    subset_noisy = df_noisy[df_noisy['Temp'] == temp]
    if len(subset_noisy) > n_samples:
        subset_noisy = subset_noisy.sample(n=n_samples, random_state=42)
    
    plt.figure(figsize=(8,6))
    
    for (idx_clean, row_clean), (idx_noisy, row_noisy) in zip(subset.iterrows(), subset_noisy.iterrows()):
        # valori puliti
        re_vals = row_clean[re_cols].values
        im_vals = row_clean[im_cols].values
        
        # valori rumorosi
        re_vals_noisy = row_noisy[re_cols].values
        im_vals_noisy = row_noisy[im_cols].values
        
        # colore basato su SoH
        soh = row_clean['SoH']
        color = plt.cm.coolwarm((soh - 80) / 20)
        # scatter pulito
        plt.scatter(re_vals, -im_vals, color=color, s=20, alpha=1, label='clean' if idx_clean==subset.index[0] else "")
        # scatter rumoroso
        plt.scatter(re_vals_noisy, -im_vals_noisy, color=color, s=50, alpha=1, marker='x', label='noisy' if idx_clean==subset.index[0] else "")


    plt.xlabel("Z' (Ohm)")
    plt.ylabel("-Z'' (Ohm)")
    plt.title(f"Comparative Nyquist Scatter - Temp {temp}°C")
    plt.grid(True)
    plt.legend()
    #plt.show()
    plt.savefig('./figures/scatter_plot_{}_percent_broadband.png'.format(int(COEFF_AMP_BROAD*100)))
    plt.close()

def main():
    
    if os.path.exists("./dataset/dataset_all_noisy_{}_percent.csv".format(int(COEFF_AMP_BROAD*100))):
        df = pd.read_csv(DATA_PATH)
        df_noisy = pd.read_csv("./dataset/dataset_all_noisy_{}_percent.csv".format(int(COEFF_AMP_BROAD*100)))
        scatter_plot(df,df_noisy)
    
    
    #if(True):
    else:
        df = pd.read_csv(DATA_PATH)
        
        re_cols, im_cols, freqs = get_sorted_impedance_cols(df.columns)

        noisy_real_list = []
        noisy_imag_list = []

        for i in range(len(df)):
            Z_real = np.array(df[re_cols].iloc[i].values).astype(float)
            Z_imag = np.array(df[im_cols].iloc[i].values).astype(float)
            Z_complex_gt = Z_real + 1j * Z_imag 
            
            print(f"Processing row {i}...") # Ridotto output per pulizia
            
            # introduco il rumore
            Z_complex_noisy = noise(Z_complex_gt, freqs)
            
        
            noisy_real_list.append(np.real(Z_complex_noisy))
            noisy_imag_list.append(np.imag(Z_complex_noisy))

        
        df_noisy_real = pd.DataFrame(noisy_real_list, columns=re_cols)
        df_noisy_imag = pd.DataFrame(noisy_imag_list, columns=im_cols)

        # aggiungo le altre colonne di condizione 
        metadata_cols = ["Cell_Name", "SoH", "SoH_Actual", "Temp", "SoC"]
        metadata = df[metadata_cols].copy()
        # per intercalare Re e Im
        impedance_cols = [c for c in df.columns if c.endswith('-Re') or c.endswith('-Im')]
        impedance_cols_ordered = []
        for c in df.columns:
            if c.endswith('-Re') or c.endswith('-Im'):
                impedance_cols_ordered.append(c)
        
        df_noisy_imp = pd.DataFrame(index=df.index)
        for c in impedance_cols_ordered:
            if c.endswith('-Re'):
                df_noisy_imp[c] = df_noisy_real[c]
            else:
                df_noisy_imp[c] = df_noisy_imag[c]
        
        df_noisy = pd.concat([metadata, df_noisy_imp], axis=1)
        df_noisy.to_csv('./dataset/dataset_all_noisy_{}_percent.csv'.format(int(COEFF_AMP_BROAD*100)), index=False)
        print("Dataset saved.")
            
    # plotto
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d') 
        
        plot_nyquist(df_noisy, ax, "NOISY DATA",samples_per_temp=127)

        # Aggiungi una colorbar comune per tutta la figura
        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=80, vmax=100))
        sm.set_array([])
        
        # CORRETTO QUI: aggiunto "ax=" prima di ax
        fig.colorbar(sm, ax=ax, label='SoH (State of Health) %', shrink=0.5, pad=0.05)

        plt.show()
if __name__ == "__main__":
    main()
