import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from model import CrossAttentionUnetModel
from conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch.optim as optim
import torch.nn as nn
filename  = '/home/users/xingyu.zhang/workspace/SD-origin/scripts/planning_train_train_all_de.pkl'
features = pickle.load(open(filename, 'rb'))
instance_features = []
map_instance_features = []

device = torch.device("cuda:5")

for i in range(len(features)):
    instance_features.append(features[i]['instance_feature'])
    map_instance_features.append(features[i]['map_instance_features'])
class FeaturesDataset(Dataset):
    def __init__(self, labels_file):
        # 读取特征和标签文件
        # with open(features_file, 'rb') as f:
        #     self.features = pickle.load(f)
        with open(labels_file, 'rb') as f:
            self.labels = pickle.load(f)
        self.instance_features = instance_features
        self.map_instance_features = map_instance_features
        
        # # 确保特征和标签长度一致
        # assert len(self.features) == len(self.labels), "Features and labels must have the same length"

    def __len__(self):
        # 返回数据的总长度
        return len(self.labels)

    def __getitem__(self, idx):
        # 根据索引返回特征和对应的标签
        instance_feature = self.instance_features[idx].to(device)
        map_instance_feature = self.map_instance_features[idx].to(device)
        trajs = self.labels[idx]['ego_trajs'].squeeze(0).to(device)
        return instance_feature, map_instance_feature,trajs


dataset = FeaturesDataset('/home/users/xingyu.zhang/workspace/SD-origin/scripts/features_train_train_all_de.pkl')

dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

print(dataloader)

model = CrossAttentionUnetModel(feature_dim=256)


train_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            trained_betas=None,
            variance_type="fixed_small",
            clip_sample=True,
            prediction_type="epsilon",
            thresholding=False,
            dynamic_thresholding_ratio=0.995,
            clip_sample_range=1.0,
            sample_max_value=1.0,
            timestep_spacing="leading",
            steps_offset=0,
            rescale_betas_zero_snr=False,
        )

def pyramid_noise_like(trajectory, discount=0.9):
    # refer to https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2?s=31
    b, n, c = trajectory.shape # EDIT: w and h get over-written, rename for a different variant!
    trajectory_reshape = trajectory.permute(0, 2, 1)
    up_sample = torch.nn.Upsample(size=(n), mode='linear')
    noise = torch.randn_like(trajectory_reshape)
    for i in range(10):
        r = torch.rand(1, device=trajectory.device) + 1  # Rather than always going 2x,
        n = max(1, int(n/(r**i)))
        # print(i, n)
        noise += up_sample(torch.randn(b, c, n).to(trajectory_reshape)) * discount**i
        if n==1: break # Lowest resolution is 1x1
    # print(noise, noise/noise.std())
    noise = noise.permute(0, 2, 1)
    return (noise/noise.std()).float()

def get_rotation_matrices(theta):
    """
    给定角度 theta，返回旋转矩阵和逆旋转矩阵

    参数:
    theta (float): 旋转角度（以弧度表示）

    返回:
    rotation_matrix (torch.Tensor): 旋转矩阵
    inverse_rotation_matrix (torch.Tensor): 逆旋转矩阵
    """
    # 将角度转换为张量
    theta_tensor = torch.tensor(theta)
    
    # 计算旋转矩阵和逆旋转矩阵
    cos_theta = torch.cos(theta_tensor)
    sin_theta = torch.sin(theta_tensor)

    rotation_matrix = torch.tensor([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])

    inverse_rotation_matrix = torch.tensor([
        [cos_theta, sin_theta],
        [-sin_theta, cos_theta]
    ])
    
    return rotation_matrix, inverse_rotation_matrix

def apply_rotation(trajectory, rotation_matrix):
    # 将 (x, y) 坐标与旋转矩阵相乘
    rotated_trajectory = torch.einsum('bij,bkj->bik', rotation_matrix, trajectory)
    return rotated_trajectory

def normalize_xy_rotation(trajectory, N=30, times=10):
        batch, num_pts, dim = trajectory.shape
        downsample_trajectory = trajectory[:, :N, :]
        x_scale = 15
        y_scale = 75
        downsample_trajectory[:, :, 0] /= x_scale
        downsample_trajectory[:, :, 1] /= y_scale
        
        rotated_trajectories = []
        for i in range(times):
            theta = 2 * torch.pi * i / 10  # 将角度均匀分布在0到2π之间
            rotation_matrix, _ = get_rotation_matrices(theta)
            # 扩展旋转矩阵以匹配批次大小
            rotation_matrix = rotation_matrix.unsqueeze(0).expand(downsample_trajectory.size(0), -1, -1).to(downsample_trajectory)
            
            rotated_trajectory = apply_rotation(downsample_trajectory, rotation_matrix)
            rotated_trajectories.append(rotated_trajectory)
        resulting_trajectory = torch.cat(rotated_trajectories, 1)
        trajectory = resulting_trajectory.permute(0,2,1)
        return trajectory



optimizer = optim.AdamW(model.parameters(), lr=5e-4)

# Define the loss function (MSE Loss for DDPM)
criterion = nn.MSELoss()


num_epochs = 100
num_points = 6
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
batch_losses = []
for epoch in range(num_epochs):
    losses = []
    model.train()
    for batch_idx, (instance_feature, map_instance_feature,trajs) in enumerate(dataloader):
        #print(batch_idx)
        batch_size = instance_feature.shape[0]
        instance_feature,map_instance_feature,trajs = instance_feature.to(device),map_instance_feature.to(device),trajs.to(device)
        device = instance_feature.device
        trajs = normalize_xy_rotation(trajs, N=num_points, times=10) 
        #print(trajs.shape)
        noise = pyramid_noise_like(trajs)
        #print(noise.shape)
        #print(trajs.shape) 
        timesteps = torch.randint(0, train_scheduler.num_train_timesteps, (batch_size,), dtype=torch.long,device=device)
        noisy_traj_points = train_scheduler.add_noise(
                original_samples=trajs,
                noise=noise,
                timesteps=timesteps,
        ).float()
        #print(timesteps)
        noise_pred = model(instance_feature, map_instance_feature,timesteps,noisy_traj_points)
        diffusion_loss = criterion(noise_pred, noise)
        optimizer.zero_grad()
        diffusion_loss.backward()
        losses.append(diffusion_loss.item())
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: { diffusion_loss.item():.4f}")
    average_loss = sum(losses) / len(losses)
    batch_losses = []
    print(f"Average Loss: {average_loss:.4f}")
    batch_losses.append(average_loss)
    if epoch % 20 == 0:
        checkpoint_filename = f'checkpoint_loss_{average_loss}_epoch_{epoch}_0903.pth'
        torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_filename)

print(batch_losses)
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'checkpoint_final_with_map.pth')
