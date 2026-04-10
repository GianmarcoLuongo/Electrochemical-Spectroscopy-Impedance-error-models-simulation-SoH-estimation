"""
Script to compute Kramers Kronig transformations starting from real measurements
"""

import numpy as np
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import impedance.validation as iv
import numpy as np

from impedance.validation import linKK
from impedance.visualization import plot_nyquist, plot_residuals
from mpl_toolkits.mplot3d import Axes3D


DATA_PATH = '/home/skywalk3r/Scrivania/SoH_GPR_customKernel/dataset/dataset_all.csv'
cols_to_drop = ['Cell_Name', 'SoH', 'SoH_Actual', 'Temp', 'SoC']

def plot_3d_nyquist_comparison(df_orig, df_fit, samples_per_temp=5):

    """
    Plot 3D Nyquist:
    - Asse X: Temperatura
    - Asse Y: Z Real
    - Asse Z: -Z Imag
    
    Confronta:
    - Punti : Dati Misurati (da df_orig)
    - Linee (Lines): Fit Lin-KK (da df_fit)
    - Colore: SoH (State of Health)
    """

    print("Generazione grafico 3D Misurato vs Fit...")
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Identifichiamo le colonne di impedenza (assumendo siano ordinate uguali in entrambi i df)
    #    Usa la tua funzione helper o logica standard:
    cols = df_orig.columns
    re_cols = [c for c in cols if 'Zreal' in c or c.endswith('-Re')] # Adatta al tuo nome colonna
    im_cols = [c for c in cols if 'Zimag' in c or c.endswith('-Im')] # Adatta al tuo nome colonna
    
    unique_temps = sorted(df_orig['Temp'].unique())
    
    # Colormap per il SoH
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=df_orig['SoH'].min(), vmax=df_orig['SoH'].max())

    for temp in unique_temps:
        # Filtriamo le righe per questa temperatura
        indices = df_orig[df_orig['Temp'] == temp].index
        
        # alcune curve prese casualmente
        if len(indices) > samples_per_temp:
            selected_indices = np.random.choice(indices, samples_per_temp, replace=False)
        else:
            selected_indices = indices
            
        for idx in selected_indices:
            # --- DATI MISURATI (df_orig) ---
            meas_re = df_orig.loc[idx, re_cols].values.astype(float)
            meas_im = df_orig.loc[idx, im_cols].values.astype(float)
            
            # --- DATI FITTATI (df_fit) ---

            fit_re = df_fit.loc[idx, re_cols].values.astype(float)
            fit_im = df_fit.loc[idx, im_cols].values.astype(float)
            
            # Asse X: Temperatura (costante per questa curva)
            temp_vals = np.full(meas_re.shape, temp)
            
            # Colore basato sul SoH di questa riga
            soh_val = df_orig.loc[idx, 'SoH']
            color = cmap(norm(soh_val))
            

            ax.plot(temp_vals, meas_re, -meas_im, 
                    linestyle='', marker='o', markersize=2, 
                    color=color, alpha=0.3)


            ax.plot(temp_vals, fit_re, -fit_im, 
                    linestyle='-', linewidth=1.5, 
                    color=color, alpha=0.9)


    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Z' Real [Ohm]")
    ax.set_zlabel("-Z'' Imag [Ohm]")
    ax.set_title("3D Nyquist Comparison: Measured (Dots) vs Lin-KK (Lines)")


    ax.view_init(elev=20, azim=-60)
    

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('SoH (State of Health) %')
    
    plt.tight_layout()
    plt.show()

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



def main():

    df = pd.read_csv(DATA_PATH)
    re_cols, im_cols, freqs = get_sorted_impedance_cols(df.columns)
    
    fit_real_list = []
    fit_imag_list = []
    residual_real_list = []
    residual_imag_list = []

    for i in range(len(df)):
        Z_real = np.array(df[re_cols].iloc[i].values).astype(float)
        Z_imag = np.array(df[im_cols].iloc[i].values).astype(float)
        Z_complex = Z_real + 1j * Z_imag 
        Z_complex = np.array(Z_complex).astype(complex)
        print("This is the complex impedance:",Z_complex)

        # droppo le colonne per prendere solo i valori di impedenza
        #df_imp = df.drop(cols_to_drop, axis=1)
        # evoco linKK
        M, mu, Z_linKK, res_real, res_imag = linKK(freqs, Z_complex, c=0.01, max_M=30, fit_type='complex', add_cap=True)
        print("M:",M)
        print("mu:",mu)
        plt.plot(Z_complex.real, -Z_complex.imag)
        plt.plot(Z_linKK.real, -Z_linKK.imag)
        
        plt.show()
        #return

        print('\nCompleted Lin-KK Fit\nM = {:d}\nmu = {:.2f}'.format(M, mu))
        
        fit_real_list.append(Z_linKK.real) 
        fit_imag_list.append(Z_linKK.imag)
        residual_real_list.append(res_real)
        residual_imag_list.append(res_imag)


    # dataframe parte reale
    df_fit_real = pd.DataFrame(fit_real_list, columns=re_cols)
    
    # dataframe parte immaginaria
    df_fit_imag = pd.DataFrame(fit_imag_list, columns=im_cols)
    # reinserisco le colonne

    metadata = df.drop(re_cols + im_cols, axis=1).reset_index(drop=True)
    df_fit_KK = pd.concat([metadata, df_fit_real, df_fit_imag], axis=1)
    df_fit_KK.to_csv('dataset_all_KK.csv', index=False)
    print("\nStruttura df_fit_KK:", df_fit_KK.shape)
    print(df_fit_KK.head)
    plot_3d_nyquist_comparison(df, df_fit_KK, samples_per_temp=127)

    res_re_matrix = np.array(residual_real_list)
    res_im_matrix = np.array(residual_imag_list)

    # ricostruisco il residuo complesso e calcolo il modulo
    Z_residual_matrix = res_re_matrix + 1j * res_im_matrix
    
    # calcolo il modulo (valore assoluto) dell'errore per ogni freq
    res_modulus_matrix = np.abs(Z_residual_matrix)

    # calcolo RANDOM DEVIATION (Std Dev del Modulo)
    random_dev_mod = np.std(res_modulus_matrix, axis=0)

    # calcolo TOTAL DEVIATION (RMS del Modulo)
    total_dev_mod = np.sqrt(np.mean(np.square(res_modulus_matrix), axis=0))


    """
    Plotto le deviazioni
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    #plot total deviation
    ax.plot(freqs, total_dev_mod, 
            label='Total Deviation (RMS)', 
            color='black', linestyle='-', linewidth=2)

    #plot random deviation
    ax.plot(freqs, random_dev_mod, 
            label='Random Deviation (Std)', 
            color='red', linestyle='--', linewidth=2)
    
    # Riempie l'area sotto la Random Deviation per evidenziare il rumore
    ax.fill_between(freqs, 0, random_dev_mod, color='red', alpha=0.1)

    ax.set_ylabel("Impedance Residual Magnitude") 
    ax.set_xlabel("Frequency [Hz]")
    ax.set_title("Analysis of Impedance Magnitude Deviation")
    ax.set_xscale('log')
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

