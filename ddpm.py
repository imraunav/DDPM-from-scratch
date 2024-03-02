import os
import torch
from tqdm import tqdm


class Diffusion:
    def __init__(
        self,
        noise_step=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=256,
        device="cuda",
    ):
        self.noise_step = noise_step
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(
            start=self.beta_start,
            end=self.beta_end,
            steps=self.noise_step,
            device=self.device,
        )

    def noise_image(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[
            :, None, None, None
        ]
        noise = torch.randn_like(x, device=self.device)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_step, size=(n,))

    @torch.no_grad()
    def sample(self, model, n, img_size=None):
        if img_size is None:
            img_size = self.img_size
        model.eval()
        x = torch.randn((n, 3, img_size, img_size), device=self.device)  # [N, C, H, W]
        for i in tqdm(reversed(range(1, self.noise_step)), position=0):
            t = torch.tensor([i] * n, dtype=torch.long).to(self.device)
            predicted_noise = model(x, t)
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = (
                1
                / torch.sqrt(alpha)
                * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise)
            ) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).to(torch.uint8)
        return x

if __name__ == "__main__":
    batch_size = 32
    diffusion = Diffusion(img_size=64, device="cpu")
    x = torch.randn((batch_size, 3, 64, 64))
    t = diffusion.sample_timesteps(batch_size)
    print(t.shape)
    x_t, noise = diffusion.noise_image(x, t)
    print(x_t.shape, noise.shape)