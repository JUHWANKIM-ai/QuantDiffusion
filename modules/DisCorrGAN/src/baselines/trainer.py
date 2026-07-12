import torch
from torch import nn
from tqdm import tqdm
import os
try:
    import wandb
except ModuleNotFoundError:               # wandb optional: no-op stub for headless benchmark runs
    class _WandbStub:
        def init(self, *a, **k):
            pass
        def log(self, *a, **k):
            pass
    wandb = _WandbStub()
from os import path as pt
from src.utils import *
import time

# trainer.py → …/DisCorrGAN/src/baselines/ — two levels up is DisCorrGAN project root (cwd-independent)
_DISCORRGAN_ROOT = pt.abspath(pt.join(pt.dirname(__file__), "..", ".."))

config_dir = pt.join("configs/config.yaml")
config = (load_config(config_dir))

def linear_rampup(current_epoch, rampup_epochs, final_value):
    init_epoch = 0
    if current_epoch < init_epoch:
        return 0.0
    elif current_epoch < rampup_epochs:
        return final_value * (current_epoch - init_epoch) / (rampup_epochs - init_epoch)
    else:
        return final_value
    
def correlation_loss_pair(
    fake_i: torch.Tensor,  # (batch, 1, time)
    fake_j: torch.Tensor,  # (batch, 1, time)
    real_i: torch.Tensor,  # (batch, 1, time)
    real_j: torch.Tensor,  # (batch, 1, time)
    corr_loss_type: str = "l2"
):
    """
    자산 쌍 (i, j)에 대해, 생성(fake)과 실제(real) 시계열 간 상관계수 차이를 계산해 반환합니다.
    - shape은 모두 (batch_size, 1, time)로 가정.
    - 반환되는 상관계수는 (2 x 2) 행렬이지만, 실제로 의미 있는 값은 off-diagonal(0,1)과 (1,0)만.
    """

    # 1) 두 자산을 합쳐서 (batch_size, 2, time) 형태로 만든다
    fake_pair = torch.cat([fake_i, fake_j], dim=1)  # (batch, 2, time)
    real_pair = torch.cat([real_i, real_j], dim=1)  # (batch, 2, time)

    # 2) corrcoef 계산을 위해 (2, batch*time) 형태로 reshape
    B, F, T = fake_pair.shape  # F=2
    fake_reshaped = fake_pair.reshape(B, F, T)  # (batch, 2, time)
    real_reshaped = real_pair.reshape(B, F, T)  # (batch, 2, time)

    # (batch, 2, time)을 corrcoef에 직접 넣을 수 없으므로, (2, B*time) 형태로 바꿔야 함.
    #   - batch 차원과 time 차원을 합쳐서 하나의 '샘플' 차원으로 사용.
    fake_2D = fake_reshaped.transpose(1, 2).reshape(-1, F).T  # (F=2, B*T)
    real_2D = real_reshaped.transpose(1, 2).reshape(-1, F).T  # (F=2, B*T)

    # 3) 2차원 텐서에 대해 corrcoef(2x2 상관행렬) 계산
    fake_corr = torch.corrcoef(fake_2D)  # (2, 2)
    real_corr = torch.corrcoef(real_2D)  # (2, 2)

    # 4) off-diagonal 원소(즉, fake_corr[0,1]와 real_corr[0,1])의 차이를 이용해 손실 계산
    #    (대각선은 자기상관이므로 보통 1에 가깝고 학습신호로 크게 의미가 없으므로 제외)
    #    필요에 따라 (0,1)과 (1,0) 양쪽을 모두 고려하거나 하나만 고려할 수 있음.
    diff = fake_corr[0, 1] - real_corr[0, 1]  # 스칼라(배치 평균 상관계수 차이)
    
    if corr_loss_type == "l2":
        loss = diff.pow(2)
    elif corr_loss_type == "l1":
        loss = diff.abs()
    elif corr_loss_type == "fro":
        # 2x2 행렬 자체의 Frobenius Norm을 쓰려면 (fake_corr - real_corr)을 통째로 쓸 수 있음.
        # 단, 여기서는 "pair" 단위로 2x2 행렬이므로, off-diagonal만 쓰고 싶다면 diff만 써야 함.
        loss = torch.norm(fake_corr - real_corr, p="fro")
    else:
        raise ValueError(f"Unknown corr_loss_type: {corr_loss_type}")
    
    return loss, fake_corr

class GANTrainer:
    def __init__(self, D, G, train_dl, config, **kwargs):                
        self.G = G
        self.D = D        
        self.config = config                                                        
        self.train_dl = train_dl        
        self.device = self.config.device

        # Directory settings
        self.full_name = (
            f'NEW_{config.seed}_{config.n_epochs}_{config.batch_size}_Glr_{config.lr_G}_Dlr_{config.lr_D}_hidden_dim_{config.hidden_dim}_n_steps_{config.n_steps}'
            f'_corr_loss_{config.corr_loss_type}_corr_weight_{config.corr_weight}_f_epoch_{config.rampup_epochs}_n_critic_{config.D_steps_per_G_step}_gp_{config.lambda_gp}_noise_{config.noise_dim}_Adam_drop_{config.G_dropout}_{config.D_dropout}'
        )
        self.results_dir = pt.join(_DISCORRGAN_ROOT, "results", "models", self.full_name)
        os.makedirs(self.results_dir, exist_ok=True)                    
        self.G_optimizer = {i: torch.optim.Adam(self.G[i].parameters(), lr=config.lr_G, betas=(0.5, 0.999)) for i in range(config.n_vars)}
        self.D_optimizer = {i: torch.optim.Adam(self.D[i].parameters(), lr=config.lr_D, betas=(0.5, 0.999)) for i in range(config.n_vars)}
        
        for i in range(config.n_vars):
            self.G[i].to(self.device)
            self.D[i].to(self.device)

    def generate_fake(self, noise):
        """
        각 생성자(generator)를 통과시켜 나온 fake sample들을 (B, n_vars, T) 형태로 concat
        """
        fake_samples = []
        for i in range(self.config.n_vars):
            # 각 TCNGenerator의 출력 shape는 (B, 1, T)로 가정
            fake_samples.append(self.G[i](noise))
        fake = torch.cat(fake_samples, dim=1)  # (B, n_vars, T)
        return fake
    
    def compute_gradient_penalty(self, discriminator, real_samples, fake_samples):
        """Compute the gradient penalty for WGAN-GP."""
        # 배치 크기, 1, 1 형태의 임의의 alpha 생성 후 real_samples 크기에 맞게 확장
        alpha = torch.rand(real_samples.size(0), 1, 1).to(self.device)
        alpha = alpha.expand_as(real_samples)

        # 샘플 간 선형 보간
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)

        # discriminator를 사용해서 interpolates에 대해 판별자 출력 계산
        d_interpolates = discriminator(interpolates).view(-1)
        fake = torch.ones(d_interpolates.size()).to(self.device)

        # 그래디언트 계산: interpolate에 대한 d_interpolates의 기울기
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = self.config.gp * ((gradient_norm - 1) ** 2).mean()
        return gradient_penalty
            
    def fit(self):        
        wandb.init(project="MVFIT_GAN", config=config)

        for epoch in tqdm(range(self.config.n_epochs)):            
            start_time = time.time()  # ⏱️ epoch 시작 시간 기록

            corr_weight = linear_rampup(epoch, config.rampup_epochs, config.corr_weight)            
     
            for n, (real) in enumerate(self.train_dl):
                real_batch = real[0].transpose(1, 2).to(self.device)
                self.step(real_batch, n, corr_weight)        
            end_time = time.time()  # ⏱️ epoch 종료 시간 기록
            epoch_time = end_time - start_time
            print(f"[Epoch {epoch}] Time per epoch: {epoch_time:.2f} seconds")  # 출력
            
            # Save model every few epochs
            if epoch % 2 == 0 and epoch >= 50:                
                for i in range(self.config.n_vars):
                    torch.save(self.G[i].state_dict(), f'{self.results_dir}/Generator_{epoch}_var_{i}.pt')                    
    
    def step(self, real, n, corr_weight):
        batch_size = real.shape[0]       
        noise = torch.randn(batch_size, self.config.noise_dim, self.config.n_steps, device=self.device)
 
            # 판별자 업데이트: 각 변수별로 개별적으로 fake 데이터를 생성하여 업데이트
        for i in range(self.config.n_vars):
            self.D_optimizer[i].zero_grad()
            real_i = real[:, i:i+1, :]
            with torch.no_grad():
                fake_i = self.G[i](noise)  # 각 변수별 생성자에서 fake sample 생성 (B, 1, T)
            real_pred = self.D[i](real_i)
            fake_pred = self.D[i](fake_i)
            loss_D = -torch.mean(real_pred) + torch.mean(fake_pred)
            
            if self.config.gp:
                gradient_penalty = self.compute_gradient_penalty(self.D[i], real_i, fake_i)
                loss_D += self.config.lambda_gp * gradient_penalty
            
            loss_D.backward()
            self.D_optimizer[i].step()
            wandb.log({f"Discriminator Loss (Var {i})": loss_D.item()})
        
        # # (B) Generator step: adversarial loss만 반영        
        if n % self.config.D_steps_per_G_step == 0:
            for i in range(self.config.n_vars):
                self.G_optimizer[i].zero_grad()                
                fake_i = self.G[i](noise)                
                adv_loss = -torch.mean(self.D[i](fake_i))
                adv_loss.backward()
                self.G_optimizer[i].step()
                wandb.log({f"Adv G Loss {i}": adv_loss.item()})
                
            # (C) Generator step: correlation loss만 반영                                                                      
            for i in range(config.n_vars):
                self.G_optimizer[i].zero_grad()
            # test noise
            noise = torch.randn(batch_size, self.config.noise_dim, self.config.n_steps, device=self.device)
            fake = self.generate_fake(noise)

            total_corr_loss = 0.0
            # 모든 변수 쌍에 대해 correlation loss를 합산 (자기 자신과의 비교 제외)
            for i in range(config.n_vars):
                for j in range(config.n_vars):
                    if i != j:
                        loss_ij, _ = correlation_loss_pair(
                            fake[:, i:i+1, :], fake[:, j:j+1, :].detach(),
                            real[:, i:i+1, :], real[:, j:j+1, :],
                            config.corr_loss_type
                        )
                        total_corr_loss += loss_ij
            loss_corr_final = corr_weight * total_corr_loss
            loss_corr_final.backward()
            for i in range(config.n_vars):
                self.G_optimizer[i].step()
            wandb.log({f"Corr Loss": total_corr_loss.item()})  
        