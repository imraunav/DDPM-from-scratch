import os
import torch
from tqdm import tqdm
from torch import nn


class Diffusion(nn.Module):
    def __init__(
        self,
        noise_step=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=256,
        channel=3,
    ):
        super().__init__()
        self.noise_step = noise_step
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        # self.device = "cuda" # device
        self.channel = channel

        self.beta = self.prepare_noise_schedule()
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(
            start=self.beta_start,
            end=self.beta_end,
            steps=self.noise_step,
        )

    def noise_image(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[
            :, None, None, None
        ]

        sqrt_alpha_hat = sqrt_alpha_hat.to(x.device)
        sqrt_one_minus_alpha_hat = sqrt_one_minus_alpha_hat.to(x.device)
        noise = torch.randn_like(x, device=x.device)

        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_step, size=(n,))

    @torch.no_grad()
    def sample(self, model, n, img_size=None, y=None, device=None):
        if device is None:
            device = "cpu"

        if img_size is None:
            img_size = self.img_size
        model.eval()
        x = torch.randn(
            (n, self.channel, img_size, img_size), device=device
        )  # [N, C, H, W]
        for i in reversed(range(1, self.noise_step)):
            t = torch.tensor([i] * n, dtype=torch.long)
            predicted_noise = model(x, t)
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]

            alpha = alpha.to(device)
            alpha_hat = alpha_hat.to(device)
            beta = beta.to(device)

            if i > 1:
                noise = torch.randn_like(x, device=device)
            else:
                noise = torch.zeros_like(x, device=device)
            # Debug
            # print(
            #     alpha.device,
            #     alpha_hat.device,
            #     beta.device,
            #     noise.device,
            #     predicted_noise.device,
            #     x.device,
            # )
            x = (
                1
                / torch.sqrt(alpha)
                * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise)
            ) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).to(torch.uint8)
        return x


if __name__ == "__main__":
    batch_size = 32
    diffusion = Diffusion(img_size=64, device="cpu")
    x = torch.randn((batch_size, 3, 64, 64))
    t = diffusion.sample_timesteps(batch_size)
    print(t.shape)
    x_t, noise = diffusion.noise_image(x, t)
    print(x_t.shape, noise.shape)
