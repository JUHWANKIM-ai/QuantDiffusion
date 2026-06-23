import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import numpy as np

class GANTrainer:
    def __init__(self, G, D, train_dl, config):
        self.G = G
        self.D = D
        self.train_dl = train_dl        
        self.config = config
        self.device = config.device

        # Move models to device
        self.G.to(self.device)
        self.D.to(self.device)

        # Optimizers
        self.G_optimizer = Adam(self.G.parameters(), lr=config.lr_G)
        self.D_optimizer = Adam(self.D.parameters(), lr=config.lr_D)        

        # Loss function
        self.criterion = nn.BCELoss()

        # Directory setup
        full_name = f'{config.file_name}_{config.n_epochs}_{config.batch_size}_G_lr_{config.lr_G}_D_lr_{config.lr_D}_noise_{config.noise_dim}_hidden_dim_{config.G_hidden_dim}'
        self.results_dir = f'./results/models/{full_name}'        
        os.makedirs(self.results_dir, exist_ok=True)

    def fit(self):
        t = tqdm(range(self.config.n_epochs))
        for epoch in t:
            for batch_idx, real in enumerate(self.train_dl):
                self.step(real[0], batch_idx)

            # Save the G periodically
            if epoch % 5 == 0:
                torch.save(self.G.state_dict(), f'{self.results_dir}/Generator_{epoch}.pt')

    def step(self, real, batch_idx):
        real = real.to(self.device).transpose(1, 2)
        batch_size, _, seq_len = real.shape

        # Generate noise
        noise = torch.randn(batch_size, self.config.noise_dim, seq_len, device=self.device)

        # Train D
        self.D.zero_grad()
        fake = self.G(noise).detach()

        real_labels = torch.ones(batch_size, device=self.device)
        fake_labels = torch.zeros(batch_size, device=self.device)

        real_output = self.D(real).view(-1)
        fake_output = self.D(fake).view(-1)

        disc_loss_real = self.criterion(real_output, real_labels)
        disc_loss_fake = self.criterion(fake_output, fake_labels)
        disc_loss = disc_loss_real + disc_loss_fake

        disc_loss.backward()
        self.D_optimizer.step()
        
        # Apply weight clipping to the Discriminator
        for param in self.D.parameters():
            param.data.clamp_(-self.config.clip, self.config.clip)

        # Train Generator
        if batch_idx % self.config.D_steps_per_G_step == 0:
            self.G.zero_grad()
            fake = self.G(noise)
            gen_output = self.D(fake).view(-1)
            gen_loss = self.criterion(gen_output, real_labels)  # Generator aims to fool D
            gen_loss.backward()
            self.G_optimizer.step()

        if batch_idx % self.config.log_interval == 0:
            print(f"Batch {batch_idx}: D Loss: {disc_loss.item()}, Generator Loss: {gen_loss.item()}")