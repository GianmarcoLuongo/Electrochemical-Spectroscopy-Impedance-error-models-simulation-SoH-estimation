import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sanitizing_utils import get_sorted_impedance_cols
from torch.utils.data import DataLoader, TensorDataset
from sanitizing_utils import data_sanitize_reg

"""
Diz. configurazione
"""
CONFIG = {
    "DATA_PATH": 'dataset_all.csv',
    "BATCH_SIZE": 128,
    "EPOCHS": 500,
    "LATENT_DIM": 8,
    "LEARNING_RATE": (1/2)*1e-2,  
    "DEVICE": 'cpu',
    "MODEL_PATH": "vae_model.pth"
}

"""
Definizione encoder VAE
"""
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

"""
Definizione decoder VAE
"""
class VAE_Decoder(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super(VAE_Decoder, self).__init__()
        
        self.fc5 = nn.Linear(latent_dim, 16)
        self.fc6 = nn.Linear(16, 32)
        self.fc7 = nn.Linear(32, 48)
        self.fc8 = nn.Linear(48, 64)
        self.fc9 = nn.Linear(64,input_dim)
        
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, z):
        h5 = self.relu(self.fc5(z))
        h6 = self.relu(self.fc6(h5))
        h7 = self.relu(self.fc7(h6))
        h8 = self.relu(self.fc8(h7))
        return self.fc9(h8) #nessuna attivazione finale per non scalare

"""
Definizione modello VAE
"""
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        
        self.encoder = VAE_Encoder(input_dim, latent_dim)
        self.decoder = VAE_Decoder(latent_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

"""
Definizione Loss Function
"""
def loss_function(recon_x, x, mu, logvar):
    # MSE loss summing over batch
    RECON = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # KL divergence 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return RECON + KLD/ x.size(0)

"""
Definizione metodo training
"""
def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for batch in dataloader:
        x = batch[0].to(CONFIG["DEVICE"])
        optimizer.zero_grad()
        recon_x, mu, logvar = model(x)
        loss = loss_function(recon_x, x, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader.dataset)

"""
Training loop
"""
def main():
    #carico i dati
    print("Loading data...")
    df = pd.read_csv(CONFIG["DATA_PATH"])
    
    #colonne EIS ordinate ordine decrescente per frequenza
    re_cols, im_cols = get_sorted_impedance_cols(df.columns)
    impedance_cols = re_cols + im_cols
    condition_cols = ['Temp', 'SoC'] 
    feature_cols = impedance_cols + condition_cols

    print(f"Selected features (NO SOH): {feature_cols[:5]} ...")
    
    #converto in array numpy per l'elaborazione
    X = df[feature_cols].values.astype(np.float32)
    print(f"Features shape: {X.shape}")
    
    #normalizzo i dati (norm. standard)
    scaler_vae = StandardScaler()
    X_scaled = scaler_vae.fit_transform(X)
    
    #converto in tensor per pytorch
    X_tensor_vae = torch.FloatTensor(X_scaled)
    input_dim = X_tensor_vae.shape[1]

    #creo il dataset
    X_tensor_vae_train = TensorDataset(X_tensor_vae)
    
    #creo il dataloader
    dataloader_vae = DataLoader(X_tensor_vae_train, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)

    #istanzio il modello
    vae = VAE(input_dim, CONFIG["LATENT_DIM"])
    #istanzio l'optimizer
    optimizer = torch.optim.Adam(vae.parameters(), lr=CONFIG["LEARNING_RATE"])
    
    #training loop
    print(f"Starting training on {CONFIG['DEVICE']} for {CONFIG['EPOCHS']} epochs...")
    loss_history = []
    for epoch in range(CONFIG['EPOCHS']):
        loss_history.append(train_epoch(vae, dataloader_vae, optimizer))
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss_history[-1]:.4f}")

    #salvo il modello
    torch.save(vae.state_dict(), CONFIG['MODEL_PATH'])
    print(f"Model saved to {CONFIG['MODEL_PATH']}")
    
    #plotto la loss
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

    
    print("Sampling from trained VAE...")
    vae.eval()
    temperature = 1

    with torch.no_grad():
        #gen
        mu, logvar = vae.encoder(X_tensor_vae)
        z = vae.reparameterize(mu, logvar)*temperature

        generated_scaled = vae.decoder(z).cpu().numpy()
        
        
        generated_data = scaler_vae.inverse_transform(generated_scaled)

        # dataframe con le colonne generate (solo feature)
        df_gen = pd.DataFrame(generated_data, columns=feature_cols)

        # sanificazione
        df_gen = data_sanitize_reg(df_gen)

        """
        NON CREO IL CSV
        questo vae è allenato su feature+temp+SoC
        quindi non ha informazioni su SoH e non è utile per fare data augmentation
        """    
        
        print("Generating Nyquist comparison plot...")
        plt.figure(figsize=(10, 8))

        
        
        real_indices = [2,6,9]
        # Stampa Temp e SoC delle tracce reali
        print("Real traces Temp/SoC:")
        for idx in real_indices:
            print(f"Index {idx}: Temp={df.iloc[idx]['Temp']}, SoC={df.iloc[idx]['SoC']}")

        # Stampa Temp e SoC delle tracce generate
        print("Generated traces Temp/SoC:")
        for idx in real_indices:
            print(f"Index {idx}: Temp={df_gen.iloc[idx]['Temp']}, SoC={df_gen.iloc[idx]['SoC']}")
        
        for i, idx in enumerate(real_indices):
            real_re = df.iloc[idx][re_cols].values
            real_im = df.iloc[idx][im_cols].values
            plt.plot(real_re, -real_im, 'b.-', alpha=0.5, label='Real' if i == 0 else "")

        
        for i, idx in enumerate(real_indices):
            gen_re = df_gen.iloc[idx][re_cols].values
            gen_im = df_gen.iloc[idx][im_cols].values
            plt.plot(gen_re, -gen_im, 'r.--', alpha=0.5, label='Generated' if i == 0 else "")

        plt.title("Nyquist Plot: Real vs Generated (VAE sanity check)")
        plt.xlabel("Z' (Real)")
        plt.ylabel("-Z'' (Imag)")
        plt.legend()
        plt.grid(True)
        plt.show()
if __name__ == "__main__":
    main()
