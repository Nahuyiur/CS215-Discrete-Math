import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
from PIL import Image

# Step 1: Generate Noisy Images
def add_gaussian_noise(image, mean=0, std=10):  
    """
    添加高斯噪声
    """
    image = np.array(image)
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

def add_salt_pepper_noise(image, salt_prob=0.005, pepper_prob=0.005): 
    """
    添加椒盐噪声
    """
    image = np.array(image)
    noisy_image = image.copy()
    salt_mask = np.random.rand(*image.shape[:2]) < salt_prob
    pepper_mask = np.random.rand(*image.shape[:2]) < pepper_prob
    noisy_image[salt_mask] = 255
    noisy_image[pepper_mask] = 0
    return Image.fromarray(noisy_image)

def generate_noisy_images(input_dir, output_dir, noise_type='gaussian'):
    """
    为所有图像生成噪声版本
    """
    os.makedirs(output_dir, exist_ok=True)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    for img_name in os.listdir(input_dir):
        if not img_name.lower().endswith(valid_extensions):
            print(f"Skipping non-image file: {img_name}")
            continue

        img_path = os.path.join(input_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Failed to process {img_name}: {e}")
            continue

        if noise_type == 'gaussian':
            noisy_image = add_gaussian_noise(image)
        elif noise_type == 'salt_pepper':
            noisy_image = add_salt_pepper_noise(image)
        else:
            raise ValueError("Invalid noise type. Choose 'gaussian' or 'salt_pepper'.")

        noisy_img_path = os.path.join(output_dir, img_name)
        noisy_image.save(noisy_img_path)
        print(f"Saved noisy image: {noisy_img_path}")

# Step 2: Define GCN Model with Residual Connections
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, adjacency_matrix, features):
        return torch.relu(adjacency_matrix @ features @ self.weight)

class GCNImageDenoising(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNImageDenoising, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)
        self.gc3 = GraphConvolution(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)  # Dropout 防止过拟合

    def forward(self, adjacency_matrix, features):
        h1 = self.dropout(self.gc1(adjacency_matrix, features))
        h2 = self.dropout(self.gc2(adjacency_matrix, h1))
        output = self.gc3(adjacency_matrix, h2)
        return output + features  # 残差连接

# Step 3: Training Process with Learning Rate Scheduler
def train_gcn(model, adjacency_matrix, noisy_image, clean_image, learning_rate=0.005, epochs=100000):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5) 
    loss_function = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        denoised_image = model(adjacency_matrix, noisy_image)
        loss = loss_function(denoised_image, clean_image)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    return denoised_image.detach()

# Step 4: Evaluate Model
def compute_metrics(clean_image, denoised_image):
    psnr_value = psnr(clean_image, denoised_image, data_range=clean_image.max() - clean_image.min())
    ssim_value = ssim(clean_image, denoised_image, multichannel=True, data_range=clean_image.max() - clean_image.min(), win_size=3)
    epi_value = np.mean(np.abs(clean_image - denoised_image))
    return psnr_value, ssim_value, epi_value

# Main Execution
if __name__ == "__main__":
    # Generate noisy images
    input_dir = 'input_dir'
    gaussian_output_dir = 'gaussian_output_dir'
    salt_pepper_output_dir = 'salt_pepper_output_dir'

    print("Generating Gaussian noisy images...")
    generate_noisy_images(input_dir, gaussian_output_dir, noise_type='gaussian')
    print("Generating Salt-Pepper noisy images...")
    generate_noisy_images(input_dir, salt_pepper_output_dir, noise_type='salt_pepper')

    # Simulate data for training
    adjacency_matrix = torch.eye(100) 
    noisy_image = torch.rand(100, 3) 
    clean_image = torch.rand(100, 3) 

    # Initialize and train the model
    model = GCNImageDenoising(input_dim=3, hidden_dim=32, output_dim=3)  
    denoised_image = train_gcn(model, adjacency_matrix, noisy_image, clean_image)

    # Compute metrics
    clean_image_np = clean_image.numpy()
    denoised_image_np = denoised_image.numpy()
    psnr_value, ssim_value, epi_value = compute_metrics(clean_image_np, denoised_image_np)

    print(f"PSNR: {psnr_value:.2f}")
    print(f"SSIM: {ssim_value:.4f}")
    print(f"EPI: {epi_value:.4f}")
