# src/gossip_wm/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import config

def reparameterize(mu, logvar):
    """
    The reparameterization trick: z = mu + std * epsilon
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

class Encoder(nn.Module):
    """
    Encodes a 64x64 image into a latent vector (mu and logvar).
    """
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1) # 64x64 -> 32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # 32x32 -> 16x16
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # 16x16 -> 8x8
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # 8x8 -> 4x4
        
        # The output of the last conv layer is 256 channels of 4x4 maps.
        # Flattened size = 256 * 4 * 4 = 4096
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1) # Flatten the tensor
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    """
    Decodes a latent vector z back into a 64x64 image.
    """
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256 * 4 * 4)
        
        # Transposed convolutions to upsample
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = x.view(-1, 256, 4, 4) # Reshape to a 4x4 image with 256 channels
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        # Use sigmoid for the final layer to output pixel values between 0 and 1
        reconstructed_img = torch.sigmoid(self.deconv4(x)) 
        return reconstructed_img

class VAE(nn.Module):
    """
    A complete Variational Autoencoder model.
    """
    def __init__(self, latent_dim=config.LATENT_DIM):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar, beta):
    """
    Calculates the VAE loss, which is a sum of reconstruction loss and KL divergence.
    recon_x: reconstructed input
    x: original input
    mu: latent mean
    logvar: latent log variance
    beta: weight for KL divergence
    """
    # Reconstruction Loss (Binary Cross-Entropy or Mean Squared Error)
    # BCE is often preferred for sigmoid outputs.
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL Divergence Loss
    # Measures how much the learned distribution (mu, logvar) diverges from a
    # standard normal distribution (mean=0, variance=1).
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_div

class TransitionModel(nn.Module):
    """
    This is the Gated Recurrent Unit (GRU) based transition model.
    It predicts the next latent state distribution (mu, logvar) given the current
    latent state and the action taken.
    """
    def __init__(self, latent_dim, action_dim, hidden_dim):
        super(TransitionModel, self).__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # The input to the GRU is the concatenation of the latent state and action
        self.gru = nn.GRU(latent_dim + action_dim, hidden_dim, batch_first=True)
        
        # The output of the GRU is projected to the parameters of the next latent state
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z, action, hidden=None):
        """
        z: current latent state, shape (batch, latent_dim) or (batch, seq_len, latent_dim)
        action: action taken, shape (batch, action_dim) or (batch, seq_len, action_dim)
        hidden: previous hidden state of the GRU
        """
        # Ensure inputs have a sequence dimension
        if z.dim() == 2:
            z = z.unsqueeze(1)
        if action.dim() == 2:
            action = action.unsqueeze(1)
            
        # Concatenate latent state and action
        z_action = torch.cat([z, action], dim=-1)
        
        # Pass through GRU
        gru_out, next_hidden = self.gru(z_action, hidden)
        
        # Get mu and logvar for the *next* latent state
        next_z_mu = self.fc_mu(gru_out)
        next_z_logvar = self.fc_logvar(gru_out)
        
        return next_z_mu, next_z_logvar, next_hidden

class WorldModel(nn.Module):
    """
    The complete World Model, combining the VAE and the Transition Model.
    """
    def __init__(self):
        super(WorldModel, self).__init__()
        self.vae = VAE(latent_dim=config.LATENT_DIM)
        self.transition = TransitionModel(
            latent_dim=config.LATENT_DIM,
            action_dim=config.ACTION_DIM,
            hidden_dim=config.TRANSITION_HIDDEN_DIM
        )

    def load_vae_weights(self, path="vae_model.pth"):
        """Utility to load pre-trained VAE weights."""
        self.vae.load_state_dict(torch.load(path, map_location=config.DEVICE))
        print(f"Successfully loaded VAE weights from {path}")