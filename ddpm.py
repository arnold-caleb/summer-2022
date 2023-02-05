import os
import torch
import time
import wandb
import numpy as np
from einops import rearrange

import torch.nn.functional as F
from torch.optim import Adam
from torchvision.utils import save_image
from torchvision import transforms, datasets

from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from models.unet import Unet
from models.resnets import exists
from diffusion.forward_diffusion import linear_beta_schedule, cosine_beta_schedule

wandb.init(project='ddpm_cardiomegaly')

data_dir = '../data_gen'
batch_size, image_size, channels = 4, 128, 1 

data_transforms = transforms.Compose([
            transforms.Resize(image_size), 
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Grayscale(channels),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),])

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
image_datasets = datasets.ImageFolder(data_dir, data_transforms)
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=True, num_workers=8)

timesteps = 700 

# beta schedule
betas = linear_beta_schedule(timesteps=timesteps)

# alphas
alphas = 1. - betas 
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# forward diffusion process q(x_t | x_{t-1}) 
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1 - alphas_cumprod_prev) / (1. - alphas_cumprod)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# forward diffusion process
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def p_losses(denoise_model, x_start, t, noise=None, loss_type='l1'):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    return F.smooth_l1_loss(noise, predicted_noise)

batch = next(iter(dataloaders))

@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    # Equation 11 in the paper
    # Use model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

results_folder = Path("./results_cardiomegaly_pa")
results_folder.mkdir(exist_ok = True)
save_and_sample_every, epochs = 1000, 300

model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4,))
model.to(device)
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
optimizer = Adam(model.parameters(), lr=1e-4)

wandb.config = {
        'timesteps': timesteps, 'learning_rate': 1e-4, 'epochs': epochs, 'batch_size': batch_size, \
        'image_size': image_size, 'channels': channels}
wandb.watch(model)

since = time.time()
for epoch in range(epochs):
    for step, batch in enumerate(dataloaders):
      optimizer.zero_grad()

      batch_size = batch[0].shape[0]
      batch = batch[0].to(device)

      # Algorithm 1 line 3: sample t uniformally for every example in the batch
      t = torch.randint(0, timesteps, (batch_size,), device=device).long()
      loss = p_losses(model, batch, t, loss_type="huber")

      if step % 100 == 0:
        print("Loss:", loss.item())

      wandb.log({'loss': loss})

      loss.backward()
      optimizer.step()

      #save generated images
      if step != 0 and step % save_and_sample_every == 0:
        milestone = step // save_and_sample_every
        batches = num_to_groups(4, batch_size)
        all_images_list = list(map(lambda n: sample(model, image_size, batch_size=n, channels=channels), batches))
        all_images = rearrange(torch.tensor(np.array(all_images_list)), "u v b c h w -> (u v b) c h w")
        #all_images = torch.cat(all_images_list, dim=0)
        all_images = (all_images + 1) * 0.5
        save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow = 6)

time_elapsed = time.time() - since
print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

torch.save(model.state_dict(), "diffusion_weights_cardiomegaly_pa.pth")
