import pandas as pd
import re
import os

def parse_input(input_file):
    rows = []
    metriche_names = ["RMSE", "MAE", "MAX_ERR", "R2"]

    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Auto-detect: usa TAB se presente, altrimenti virgola
            if "\t" in line:
                parts = line.split("\t")
            else:
                parts = line.split(",")
            
            if len(parts) != 5:
                continue

            model_name = parts[0].strip()

            # Salta header
            if model_name == "" or ("RMSE" in parts[1] and "±" not in parts[1]):
                continue

            row_data = {"Model": model_name}

            for i, m in enumerate(metriche_names):
                val_str = parts[i + 1].strip()
                # Cattura i numeri positivi, negativi e in notazione scientifica
                match = re.match(r'([+-]?[\d.]+)\s*±\s*([+-]?[\d.]+)', val_str)
                if match:
                    row_data[f"Average {m}"] = float(match.group(1))
                    row_data[f"Std {m}"] = float(match.group(2))
                else:
                    row_data[f"Average {m}"] = None
                    row_data[f"Std {m}"] = None

            rows.append(row_data)

    return pd.DataFrame(rows)


def main():
    input_file = "metriche_modelli_optimized.csv"
    output_file = "metriche_analisi_optimized.xlsx"

    if not os.path.exists(input_file):
        print(f"Errore: Il file '{input_file}' non esiste.")
        return

    df = parse_input(input_file)
    print(f"Modelli caricati: {len(df)}")
    print(f"Colonne: {list(df.columns)}")

    metriche = ["RMSE", "MAE", "MAX_ERR", "R2"]

    # =========================================================
    # MAPPING NOMI INCONSISTENTI
    # =========================================================
    name_mapping = {
        "KNN_regressor": "KNN",
        "vae_enc_reg_head": "VAE_regressor_head",
        "Kernel_SVR": "KernelSVR",
    }

    # =========================================================
    # FORMAT PAPER TABLE
    # =========================================================
    def format_paper_table(dataframe):
        if dataframe.empty:
            return pd.DataFrame()
        df_out = pd.DataFrame()
        df_out["Model"] = dataframe["Model"].values
        for m in metriche:
            avg_col = f"Average {m}"
            std_col = f"Std {m}"
            if avg_col in dataframe.columns and std_col in dataframe.columns:
                df_out[m] = (
                    dataframe[avg_col].round(5).astype(str).values
                    + " ± "
                    + dataframe[std_col].round(5).astype(str).values
                )
        return df_out

    # =========================================================
    # ESTRAZIONE BASE MODEL + SYSTEMATIC ERROR LEVEL
    # =========================================================
    def extract_meta(name):
        # Legge il formato: GPR_clean oppure GPR_noisy_0_3
        if name.endswith("_clean"):
            base = name.replace("_clean", "")
            level = "Clean"
        elif "_noisy_" in name:
            parts = name.split("_noisy_")
            base = parts[0]
            # Trasforma 0_3 in 0.3 per poterlo trattare come numero float
            level = parts[1].replace('_', '.')
        else:
            base = name
            level = "Clean" # Default fallback

        # Applica mapping se il nome base è scritto in modo diverso
        base = name_mapping.get(base, base)
        return base, level

    # Creazione della colonna formale per l'errore sistematico
    df[["Base_Model", "Systematic_Instrument_Error"]] = df["Model"].apply(
        lambda x: pd.Series(extract_meta(x))
    )

    # =========================================================
    # SEPARAZIONE CLEAN / ERROR
    # =========================================================
    df_clean_raw = df[df["Systematic_Instrument_Error"] == "Clean"].copy()
    df_sys_err_raw = df[df["Systematic_Instrument_Error"] != "Clean"].copy()

    # Ordina i livelli di errore matematicamente (0.3, 1.0, 3.0, 10.0, 15.0, 20.0)
    sys_error_levels = sorted(df_sys_err_raw["Systematic_Instrument_Error"].unique(), key=lambda x: float(x))

    print(f"\nModelli clean (baseline): {len(df_clean_raw)}")
    print(f"Modelli con Errore Sistematico: {len(df_sys_err_raw)}")
    print(f"Livelli di errore rilevati: {sys_error_levels}")

    sheet_clean = format_paper_table(df_clean_raw)

    sheets_sys_err = {}
    for level in sys_error_levels:
        sheets_sys_err[level] = format_paper_table(
            df_sys_err_raw[df_sys_err_raw["Systematic_Instrument_Error"] == level]
        )

    # =========================================================
    # WORSENING VS CLEAN (Peggioramento rispetto alla baseline)
    # =========================================================
    clean_dict = df_clean_raw.set_index("Base_Model").to_dict("index")

    delta_rows = []

    for _, row in df_sys_err_raw.iterrows():
        base_name = row["Base_Model"]
        err_level = row["Systematic_Instrument_Error"]

        if base_name in clean_dict:
            base_stats = clean_dict[base_name]
            delta_data = {"Model": base_name, "Systematic_Instrument_Error": err_level}

            for m in metriche:
                col_avg = f"Average {m}"
                val_err = row[col_avg]
                val_clean = base_stats[col_avg]

                # R2 si valuta come differenza assoluta, gli errori come % di peggioramento
                if m != "R2":
                    if val_clean != 0:
                        change = ((val_err - val_clean) / abs(val_clean)) * 100
                    else:
                        change = 0
                    delta_data[f"{m}_Worsening_%"] = round(change, 4)
                else:
                    change = val_err - val_clean
                    delta_data[f"{m}_Diff_Abs"] = round(change, 6)

            delta_rows.append(delta_data)
        else:
            print(f"[WARN] Base model '{base_name}' non trovato tra i clean per il confronto!")

    df_delta = pd.DataFrame(delta_rows)

    # Ordinamento corretto della tabella Worsening basato sul valore float dell'errore
    if not df_delta.empty:
        df_delta["Error_float"] = df_delta["Systematic_Instrument_Error"].astype(float)
        df_delta = df_delta.sort_values(by=["Model", "Error_float"]).drop(columns=["Error_float"])

    # =========================================================
    # TRADEOFF = worst error (ultimo livello testato)
    # =========================================================
    df_tradeoff = pd.DataFrame()

    if sys_error_levels and not df_delta.empty:
        worst_level = sys_error_levels[-1] # Prende il livello numerico più alto (es. 20.0)

        df_delta_worst = df_delta[df_delta["Systematic_Instrument_Error"] == worst_level].copy().reset_index(drop=True)
        df_sys_err_worst = df_sys_err_raw[df_sys_err_raw["Systematic_Instrument_Error"] == worst_level].copy().reset_index(drop=True)

        if not df_delta_worst.empty and "RMSE_Worsening_%" in df_delta_worst.columns:
            # Allinea per Base_Model
            df_merge = df_delta_worst[["Model", "RMSE_Worsening_%"]].merge(
                df_sys_err_worst[["Base_Model", "Average RMSE"]],
                left_on="Model",
                right_on="Base_Model",
                how="inner"
            )

            if not df_merge.empty:
                df_merge["Rank_Worsening"] = df_merge["RMSE_Worsening_%"].rank()
                df_merge["Rank_RMSE"] = df_merge["Average RMSE"].rank()
                df_merge["Tradeoff_score"] = df_merge["Rank_Worsening"] + df_merge["Rank_RMSE"]

                df_tradeoff = df_merge[["Model", "RMSE_Worsening_%", "Average RMSE", "Tradeoff_score"]]
                df_tradeoff = df_tradeoff.sort_values("Tradeoff_score").reset_index(drop=True)

    # =========================================================
    # EXPORT EXCEL
    # =========================================================
    with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
        # 1. Foglio modelli Puliti
        sheet_clean.to_excel(writer, sheet_name="Clean_Models", index=False)

        # 2. Un foglio per ogni livello di errore sistematico
        for level in sys_error_levels:
            # Rimettiamo l'underscore per il nome del tab Excel (es. Sys_Error_0_3)
            sheet_name = f"Sys_Error_{level.replace('.', '_')}"
            sheets_sys_err[level].to_excel(writer, sheet_name=sheet_name, index=False)

        # 3. Foglio riassuntivo del peggioramento
        if not df_delta.empty:
            df_delta.to_excel(writer, sheet_name="Worsening_vs_Clean", index=False)

        # 4. Foglio Score finale
        if not df_tradeoff.empty:
            df_tradeoff.to_excel(writer, sheet_name="Tradeoff_Score", index=False)

    print(f"\n✅ Analisi completata! File Excel salvato: {output_file}")


if __name__ == "__main__":
    main()
