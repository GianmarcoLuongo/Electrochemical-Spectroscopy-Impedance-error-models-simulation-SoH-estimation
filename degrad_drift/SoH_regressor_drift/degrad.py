import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import re


DATA_PATH = 'dataset_all.csv'
DRIFT_COEFFICIENTS = [0.1e-3,0.3e-3,0.5e-3,0.7e-3,1e-3]

#estraggo freq con regex
def extract_freq(col_name):
    match = re.match(r"([0-9\.]+)-", col_name)
    return float(match.group(1)) if match else 0
    
def get_sorted_impedance_cols(columns):
    #trovo colonne che finiscono con -Re o -Im
    re_cols = [c for c in columns if c.endswith('-Re')]
    im_cols = [c for c in columns if c.endswith('-Im')]
    
    # ordino decrescente (da alta a bassa frequenza per Nyquist)
    re_cols.sort(key=extract_freq, reverse=True)
    im_cols.sort(key=extract_freq, reverse=True)
    
    # estraggo anche il vettore delle frequenze numeriche
    freqs = np.array([extract_freq(c) for c in re_cols])
    
    return re_cols, im_cols, freqs


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
   
def noise(Z_complex_gt, drift_coefficient, freqs):
    """
    Implementa il drift osservato alle basse frequenze con lo slope dovuto al drift_coefficient
    Drift dovuto a variazione dello SoC oppure drift strumentale
    L'ampiezza massima dell'errore è data dal coeff. moltiplicato per 0.01Hz
    L'errore è sistematico
    """
    Z_abs = np.abs(Z_complex_gt)
    Z_drift = np.zeros_like(Z_complex_gt, dtype=complex) 

    for i in range(len(Z_drift)):
        #calcolo ampiezza del rumore
        err_amp = drift_coefficient * Z_abs[i] * 1/freqs[i]
        #calcolo ampiezze errore reali e immaginarie scalate di 0.7071
        #avessi aggiunto rumore senza normalizzare avrei avuto modulo totale dell'errore sqrt(2)*err_amp desiderata
        err_real = err_amp / np.sqrt(2)
        err_imag = err_real
        #somma degli scostamenti 
        Z_drift[i] = err_real + 1j * err_imag


    Z_complex_noisy = Z_complex_gt + Z_drift
    return Z_complex_noisy


def scatter_plot(df, df_noisy, dc):

    """
    Scatter plot acquisizioni
    """
    
    cols = df.columns
    re_cols = [c for c in cols if c.endswith('-Re')]
    im_cols = [c for c in cols if c.endswith('-Im')]
    freqs = np.array([extract_freq(c) for c in re_cols])
    
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
        # annota la frequenza accanto ai punti puliti
        for x, y, f in zip(re_vals, -im_vals, freqs):
            plt.text(x, y, f"{f:.2g}", fontsize=8, color='gray', alpha=0.7)

    plt.xlabel("Z' (Ohm)")
    plt.ylabel("-Z'' (Ohm)")
    plt.title(f"Comparative Nyquist Scatter - Temp {temp}°C - Drift Coefficient: {dc}")
    plt.grid(True)
    plt.legend()
    #plt.show()
    plt.savefig('./figures/scatter_plot_{}_drift_coefficient.png'.format(str(dc).replace('.', '_')))
    plt.close()

def main():
    
    df = pd.read_csv(DATA_PATH)
    re_cols, im_cols, freqs = get_sorted_impedance_cols(df.columns)

    for dc in DRIFT_COEFFICIENTS:
        
        tag = f"{dc}".replace('.', '_')
        out_path = f"./dataset/dataset_all_noisy_{tag}.csv"
        
        print(f"\n--- Drift coefficient: {dc} ---")
    
        if os.path.exists(out_path):
            df_noisy = pd.read_csv(out_path)
            scatter_plot(df, df_noisy, dc)
            continue
    
        np.random.seed(42)

        noisy_real_list = []
        noisy_imag_list = []
        
        for i in range(len(df)):
            Z_real = np.array(df[re_cols].iloc[i].values).astype(float)
            Z_imag = np.array(df[im_cols].iloc[i].values).astype(float)
            Z_complex_gt = Z_real + 1j * Z_imag 
            
            print(f"Processing row {i}...")
            
            # introduco il rumore
            Z_complex_noisy = noise(Z_complex_gt,dc,freqs)
            
            noisy_real_list.append(np.real(Z_complex_noisy))
            noisy_imag_list.append(np.imag(Z_complex_noisy))

        
        df_noisy_real = pd.DataFrame(noisy_real_list, columns=re_cols)
        df_noisy_imag = pd.DataFrame(noisy_imag_list, columns=im_cols)
        metadata_cols = ["Cell_Name", "SoH", "SoH_Actual", "Temp", "SoC"]
        metadata = df[metadata_cols].copy()

        # Intercalo Re-Im in modo matematicamente corretto
        impedance_cols_ordered = []
        for re_c, im_c in zip(re_cols, im_cols):
            impedance_cols_ordered.append(re_c)
            impedance_cols_ordered.append(im_c)

        df_noisy_imp = pd.DataFrame(index=df.index)

        for re_c, im_c in zip(re_cols, im_cols):
            df_noisy_imp[re_c] = df_noisy_real[re_c]
            df_noisy_imp[im_c] = df_noisy_imag[im_c]

        df_noisy = pd.concat([metadata, df_noisy_imp], axis=1)
        print(df_noisy.head())
        print("Saving to:", out_path)
        df_noisy.to_csv(out_path, index=False)
        # plotto
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d') 
        
        plot_nyquist(df_noisy, ax, f"NOISY DATA - drift coefficient: {dc}", samples_per_temp=127)

        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=80, vmax=100))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='SoH (State of Health) %', shrink=0.5, pad=0.05)

        #plt.show()
        scatter_plot(df, df_noisy, dc)


if __name__ == "__main__":
    main()
