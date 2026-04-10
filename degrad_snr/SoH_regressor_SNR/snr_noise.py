import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import re


DATA_PATH = '/home/skywalk3r/Scrivania/SoH_GPR_customKernel/dataset/dataset_all.csv'
I_EXC_AMP = 10e-3  # 10 mA corrente di eccitazione COSINUSOIDALE per avere il fasore di corrente a sola parte reale 
SNR_VALUES_DB = [50,40,30,20]

# estraggo freq con regex
def extract_freq(col_name):
    match = re.match(r"([0-9\.]+)-", col_name)
    return float(match.group(1)) if match else 0
    
def get_sorted_impedance_cols(columns):
    re_cols = [c for c in columns if c.endswith('-Re')]
    im_cols = [c for c in columns if c.endswith('-Im')]
    re_cols.sort(key=extract_freq, reverse=True)
    im_cols.sort(key=extract_freq, reverse=True)
    freqs = np.array([extract_freq(c) for c in re_cols])
    return re_cols, im_cols, freqs


def plot_nyquist(df, ax, title, samples_per_temp=127):
    unique_temps = sorted(df['Temp'].unique())
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
            temp_vals = np.full(re_vals.shape, temp)
            soh = row['SoH']
            color = plt.cm.coolwarm((soh - 80) / 20)
            ax.plot(temp_vals, re_vals, -im_vals, color=color, alpha=0.6, marker='o', markersize=1)

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Z' (Real)")
    ax.set_zlabel("-Z'' (Imaginary)")
    ax.set_title(title)


def noise_snr(Z_complex_gt, snr_db, I_exc):
    """
    Simula rumore realistico (AWGN) sui sensori di tensione e corrente nel dominio della frequenza.
    Inietto una corrente di ampiezza costante (10mA) in base al circuito basato su AD5941
    La corrente iniettata è cosinusoidale per avere una corrente a valori reali di amp 10mA
    
    """
    N = len(Z_complex_gt)
    
    #ricostruisco V (prendo i valori di picco)
    V_true = I_exc * Z_complex_gt
    
    #estraggo il rms di V
    #V_rms = np.sqrt(np.mean(np.abs(V_true)**2))
    V_rms = np.abs(V_true)/np.sqrt(2)
    I_rms = np.abs(I_exc)/np.sqrt(2) # perchè l'ampiezza max del forzamento in corrente è 10mA
    
    #calcolo le deviazioni in base ai valori snr in naturali
    #l'SNR calcolato sui segnali è 20log10(Xrms/sigma_x)
    sigma_V = V_rms / (10 ** (snr_db / 20))
    sigma_I = I_rms / (10 ** (snr_db / 20))
    
    # rumore gaussiano complesso
    # i rumori hanno potenza normalizzata a sigma**2 e tale potenza è ripartita in parti uguali
    # le v.a. hanno varianza pari a (sigma**2)/2
    noise_V = (sigma_V / np.sqrt(2)) * (np.random.randn(N) + 1j * np.random.randn(N))
    noise_I = (sigma_I / np.sqrt(2)) * (np.random.randn(N) + 1j * np.random.randn(N))
    
    # introduco rumore sui segnali
    V_meas = V_true + noise_V
    I_meas = I_exc + noise_I
    
    # calcolo imp misurata
    Z_meas = V_meas / I_meas
    
    return Z_meas


def scatter_plot(df, df_noisy, snr_db):
    cols = df.columns
    re_cols = [c for c in cols if c.endswith('-Re')]
    im_cols = [c for c in cols if c.endswith('-Im')]
    freqs = np.array([extract_freq(c) for c in re_cols])
    
    n_samples = 1
    temp = 15
    subset = df[df['Temp'] == temp]
    if len(subset) > n_samples:
        subset = subset.sample(n=n_samples, random_state=42)

    subset_noisy = df_noisy[df_noisy['Temp'] == temp]
    if len(subset_noisy) > n_samples:
        subset_noisy = subset_noisy.sample(n=n_samples, random_state=42)
    
    plt.figure(figsize=(8, 6))
    
    for (idx_clean, row_clean), (idx_noisy, row_noisy) in zip(subset.iterrows(), subset_noisy.iterrows()):
        re_vals = row_clean[re_cols].values
        im_vals = row_clean[im_cols].values
        re_vals_noisy = row_noisy[re_cols].values
        im_vals_noisy = row_noisy[im_cols].values
        
        soh = row_clean['SoH']
        color = plt.cm.coolwarm((soh - 80) / 20)
        
        plt.scatter(re_vals, -im_vals, color=color, s=20, alpha=1, 
                    label='clean' if idx_clean == subset.index[0] else "")
        plt.scatter(re_vals_noisy, -im_vals_noisy, color=color, s=50, alpha=1, marker='x', 
                    label='noisy' if idx_clean == subset.index[0] else "")
        
        for x, y, f in zip(re_vals, -im_vals, freqs):
            plt.text(x, y, f"{f:.2g}", fontsize=8, color='gray', alpha=0.7)

    plt.xlabel("Z' (Ohm)")
    plt.ylabel("-Z'' (Ohm)")
    plt.title(f"Comparative Nyquist Scatter - Temp {temp}°C - SNR: {snr_db} dB")
    plt.grid(True)
    plt.legend()
    plt.savefig(f'./figures/scatter_plot_snr_{snr_db}dB.png')
    plt.close()


def main():
    
    df = pd.read_csv(DATA_PATH)
    re_cols, im_cols, freqs = get_sorted_impedance_cols(df.columns)
    
    os.makedirs('./figures', exist_ok=True)
    os.makedirs('./dataset', exist_ok=True)

    for snr_db in SNR_VALUES_DB:
        
        tag = f"snr_{snr_db}dB"
        out_path = f"./dataset/dataset_all_noisy_{tag}.csv"
        
        print(f"\n--- SNR: {snr_db} dB (I_exc_peak_amp = {I_EXC_AMP*1000:.1f} mA) ---")
    
        if os.path.exists(out_path):
            df_noisy = pd.read_csv(out_path)
            scatter_plot(df, df_noisy, snr_db)
            continue
    
        np.random.seed(42)

        noisy_real_list = []
        noisy_imag_list = []
        
        for i in range(len(df)):
            Z_real = np.array(df[re_cols].iloc[i].values).astype(float)
            Z_imag = np.array(df[im_cols].iloc[i].values).astype(float)
            Z_complex_gt = Z_real + 1j * Z_imag 
            
            if i % 50 == 0:
                print(f"Processing row {i}/{len(df)}...")
            
            Z_complex_noisy = noise_snr(Z_complex_gt, snr_db, I_EXC_AMP)
            
            noisy_real_list.append(np.real(Z_complex_noisy))
            noisy_imag_list.append(np.imag(Z_complex_noisy))

        df_noisy_real = pd.DataFrame(noisy_real_list, columns=re_cols)
        df_noisy_imag = pd.DataFrame(noisy_imag_list, columns=im_cols)
        metadata_cols = ["Cell_Name", "SoH", "SoH_Actual", "Temp", "SoC"]
        metadata = df[metadata_cols].copy()

        # Intercalo Re-Im senza frammentazione
        frames = []
        for re_c, im_c in zip(re_cols, im_cols):
            frames.append(df_noisy_real[[re_c]])
            frames.append(df_noisy_imag[[im_c]])
        df_noisy_imp = pd.concat(frames, axis=1)

        df_noisy = pd.concat([metadata, df_noisy_imp], axis=1)
        
        # Verifico SNR effettivo
        Z_abs_all = np.sqrt(df[re_cols].values**2 + df[im_cols].values**2)
        noise_re_all = df_noisy_real.values - df[re_cols].values
        noise_im_all = df_noisy_imag.values - df[im_cols].values
        noise_abs_all = np.sqrt(noise_re_all**2 + noise_im_all**2)
        snr_eff = 20 * np.log10(np.mean(Z_abs_all) / np.mean(noise_abs_all))
        print(f"SNR effettivo medio: {snr_eff:.1f} dB")
        
        print("Saving to:", out_path)
        df_noisy.to_csv(out_path, index=False)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d') 
        plot_nyquist(df_noisy, ax, f"NOISY DATA - SNR: {snr_db} dB", samples_per_temp=127)
        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=80, vmax=100))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='SoH (State of Health) %', shrink=0.5, pad=0.05)

        scatter_plot(df, df_noisy, snr_db)


if __name__ == "__main__":
    main()
