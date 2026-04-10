import pandas as pd
import numpy as np
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sanitizing_utils import get_sorted_impedance_cols
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import root_mean_squared_error, max_error
from sklearn.metrics import mean_absolute_error, r2_score

np.random.seed(42)
torch.manual_seed(42)

# ========================================
# PARAMETRI DA TERMINALE
# ========================================
DATA_PATH = sys.argv[1]
LABEL = sys.argv[2]

print(f"Dataset: {DATA_PATH}")
print(f"Label: {LABEL}")

CONFIG = {
    "BATCH_SIZE": 32,
    "EPOCHS": 500,
    "LATENT_DIM": 8,
    "LEARNING_RATE": 0.5e-3,
    "DEVICE": 'cpu',
    "MODEL_PATH": "vae_model.pth"
}

# ========================================
# DEFINIZIONE CLASSI
# ========================================
class VAE_Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE_Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 48)
        self.fc3 = nn.Linear(48, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        h4 = self.relu(self.fc4(h3))
        return self.fc_mu(h4), self.fc_logvar(h4)

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = VAE_Encoder(input_dim, latent_dim)
        self.dummy_decoder = nn.Linear(1, 1)

class RegressorHead(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(RegressorHead, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 2*latent_dim)
        self.bn1 = nn.BatchNorm1d(2*latent_dim)
        self.fc2 = nn.Linear(2*latent_dim, 2*latent_dim)
        self.bn2 = nn.BatchNorm1d(2*latent_dim)
        self.fc3 = nn.Linear(2*latent_dim, 2*latent_dim)
        self.bn3 = nn.BatchNorm1d(2*latent_dim)
        self.fc4 = nn.Linear(2*latent_dim, latent_dim)
        self.bn4 = nn.BatchNorm1d(latent_dim)
        self.fc_out = nn.Linear(latent_dim, 1)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.act(self.bn1(self.fc1(x)))
        x = self.act(self.bn2(self.fc2(x)))
        x = self.act(self.bn3(self.fc3(x)))
        x = self.act(self.bn4(self.fc4(x)))
        x = self.fc_out(x)
        return x

class VAEEncoderRegressor(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAEEncoderRegressor, self).__init__()
        self.encoder = VAE_Encoder(input_dim, latent_dim)
        self.regressor = RegressorHead(input_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.regressor(z)

# ========================================
# CV GENERATOR
# ========================================
def CVGenerator(dataset):
    dataset = dataset.dropna(subset=['SoH'])
    folds = [
        [28, 29, 2, 12, 17, 15],
        [3, 4, 13, 20, 18, 30],
        [17, 21, 14, 19, 31, 5],
        [15, 23, 24, 32, 6, 22]
    ]
    for batt_test in folds:
        trainingSet = dataset[~dataset['Cell_Name'].isin(batt_test)]
        testSet = dataset[dataset['Cell_Name'].isin(batt_test)]
        yield (trainingSet.index.values.astype(int), testSet.index.values.astype(int))

# ========================================
# MAIN
# ========================================
def main():
    df = pd.read_csv(DATA_PATH)

    re_cols, im_cols = get_sorted_impedance_cols(df.columns)
    impedance_cols = re_cols + im_cols
    feature_cols = impedance_cols + ['Temp', 'SoC']

    X = df[feature_cols].values.astype(np.float32)

    scaler_vae = StandardScaler()
    X_scaled = scaler_vae.fit_transform(X)
    X_tensor_vae = torch.tensor(X_scaled, dtype=torch.float32)
    input_dim = X_tensor_vae.shape[1]
    print(f"Input dimension: {input_dim}")

    print("Loading pretrained VAE weights...")
    saved_state_dict = torch.load(CONFIG["MODEL_PATH"])

    criterion = nn.HuberLoss()

    MAEs, R2s, RMSEs, Max_Errors, Losses, Test_Losses = [], [], [], [], [], []

    for fold, (train_index, test_index) in enumerate(CVGenerator(df)):
        X_train, X_test = X_tensor_vae[train_index], X_tensor_vae[test_index]
        y_train = df.loc[train_index, 'SoH_Actual'].values.astype(np.float32) / 100
        y_test = df.loc[test_index, 'SoH_Actual'].values.astype(np.float32) / 100

        y_train_mean = np.mean(y_train)
        y_train_tensor = torch.tensor(y_train - y_train_mean, dtype=torch.float32).unsqueeze(1)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

        dataloader_train = DataLoader(TensorDataset(X_train, y_train_tensor), batch_size=CONFIG["BATCH_SIZE"], shuffle=True, drop_last=True)
        dataloader_test = DataLoader(TensorDataset(X_test, y_test_tensor - y_train_mean), batch_size=CONFIG["BATCH_SIZE"], shuffle=False, drop_last=False)

        loss_history = []
        test_loss_history = []

        model = VAEEncoderRegressor(input_dim, CONFIG["LATENT_DIM"])

        encoder_dict = {k.replace('encoder.', ''): v for k, v in saved_state_dict.items() if k.startswith('encoder.')}
        model.encoder.load_state_dict(encoder_dict)
        print(f"Fold {fold+1}: Encoder weights loaded.")

        for param in model.encoder.parameters():
            param.requires_grad = False

        model.to(CONFIG["DEVICE"])
        optimizer = optim.Adam(model.regressor.parameters(), lr=CONFIG["LEARNING_RATE"])

        for epoch in range(CONFIG["EPOCHS"]):
            model.train()
            running_loss = 0.0
            running_test_loss = 0.0

            for batch in dataloader_train:
                x_batch = batch[0].to(CONFIG["DEVICE"])
                y_batch = batch[1].to(CONFIG["DEVICE"])
                optimizer.zero_grad()
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(dataloader_train)
            loss_history.append(epoch_loss)

            model.eval()
            with torch.no_grad():
                for batch in dataloader_test:
                    x_batch = batch[0].to(CONFIG["DEVICE"])
                    y_batch = batch[1].to(CONFIG["DEVICE"])
                    y_pred = model(x_batch)
                    test_loss = criterion(y_pred, y_batch)
                    running_test_loss += test_loss.item()

            test_epoch_loss = running_test_loss / len(dataloader_test)
            test_loss_history.append(test_epoch_loss)

        Losses.append(loss_history)
        Test_Losses.append(test_loss_history)

        model.eval()
        X_test_device = X_test.to(CONFIG["DEVICE"])
        with torch.no_grad():
            y_pred_test = model(X_test_device).cpu().numpy().squeeze() + y_train_mean

        MAEs.append(mean_absolute_error(y_test, y_pred_test))
        R2s.append(r2_score(y_test, y_pred_test))
        RMSEs.append(root_mean_squared_error(y_test, y_pred_test))
        Max_Errors.append(max_error(y_test, y_pred_test))

        print(f"Fold {fold+1}: MAE={MAEs[-1]:.4f}, RMSE={RMSEs[-1]:.4f}, R2={R2s[-1]:.4f}")

    # ========================================
    # RISULTATI E SALVATAGGIO
    # ========================================
    print("\n" + "="*50)
    print("RISULTATI FINALI")
    print("="*50)
    print(f"Average RMSE: {np.mean(RMSEs):.4f} ± {np.std(RMSEs):.4f}")
    print(f"Average MAE:  {np.mean(MAEs):.4f} ± {np.std(MAEs):.4f}")
    print(f"Average MAX:  {np.mean(Max_Errors):.4f} ± {np.std(Max_Errors):.4f}")
    print(f"Average R2:   {np.mean(R2s):.4f} ± {np.std(R2s):.4f}")

    metriche_finali = {
        'RMSE': f"{np.mean(RMSEs):.4f} ± {np.std(RMSEs):.4f}",
        'MAE': f"{np.mean(MAEs):.4f} ± {np.std(MAEs):.4f}",
        'MAX_ERR': f"{np.mean(Max_Errors):.4f} ± {np.std(Max_Errors):.4f}",
        'R2': f"{np.mean(R2s):.4f} ± {np.std(R2s):.4f}",
    }

    file_name = "metriche_modelli_optimized.csv"
    row_name = f"VAE_regressor_head_{LABEL}"

    if os.path.exists(file_name):
        df_existing = pd.read_csv(file_name, index_col=0)
        df_existing.loc[row_name] = metriche_finali
        df_existing.to_csv(file_name)
    else:
        pd.DataFrame([metriche_finali], index=[row_name]).to_csv(file_name, index=True)

    print(f"\nMetriche salvate come '{row_name}' in {file_name}")

    # Plot losses
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i in range(4):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        ax.plot(Losses[i], label='Train', color='blue')
        ax.plot(Test_Losses[i], label='Test', color='red', linestyle='--')
        ax.set_title(f'Fold {i+1}')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Huber Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"losses_vae_enc_reg_head_{LABEL}.png")

if __name__ == "__main__":
    main()