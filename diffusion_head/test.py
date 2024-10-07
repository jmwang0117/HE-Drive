import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from model import CrossAttentionUnetModel
from conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch.optim as optim
import torch.nn as nn
filename  = '/home/users/xingyu.zhang/workspace/SD-origin/scripts/planning_eval_all_de_with_anchor.pkl'
features = pickle.load(open(filename, 'rb'))
instance_features = []
from tqdm import tqdm
map_instance_features = []

device = torch.device("cuda:6")

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
        
        # # 确保特征和标签长度一致
        # assert len(self.features) == len(self.labels), "Features and labels must have the same length"

    def __len__(self):
        # 返回数据的总长度
        return len(self.labels)

    def __getitem__(self, idx):
        # 根据索引返回特征和对应的标签
        instance_feature = instance_features[idx].to(device)
        map_instance_feature = map_instance_features[idx].to(device)
        trajs = self.labels[idx]['ego_trajs'].to(device)
        return instance_feature, map_instance_feature,trajs


dataset = FeaturesDataset('/home/users/xingyu.zhang/workspace/SD-origin/scripts/features_eval_all_de_with_anchor.pkl')

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

model = CrossAttentionUnetModel(feature_dim=256)

checkpoint = torch.load('/home/users/xingyu.zhang/workspace/SparseDrive/diffusion_head/checkpoint_loss_0.005207525296216111_epoch_80_0903.pth')

model.load_state_dict(checkpoint['model_state_dict'])
inferece_scheduler = DDPMScheduler(
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
        x_scale = 10
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


def denormalize_xy_rotation(trajectory, N=30, times=10):
        batch, num_pts, dim = trajectory.shape
        inverse_rotated_trajectories = []
        for i in range(times):
            theta = 2 * torch.pi * i / 10  # 将角度均匀分布在0到2π之间
            rotation_matrix, inverse_rotation_matrix = get_rotation_matrices(theta)
            # 扩展旋转矩阵以匹配批次大小
            inverse_rotation_matrix = inverse_rotation_matrix.unsqueeze(0).expand(trajectory.size(0), -1, -1).to(trajectory)
        
            # 只对每个 2D 坐标对进行逆旋转
            inverse_rotated_trajectory = apply_rotation(trajectory[:, :, 2*i:2*i+2], inverse_rotation_matrix)
            inverse_rotated_trajectories.append(inverse_rotated_trajectory)

        final_trajectory = torch.cat(inverse_rotated_trajectories, 1).permute(0,2,1)
        
        final_trajectory = final_trajectory[:, :, :2]
        final_trajectory[:, :, 0] *= 13
        final_trajectory[:, :, 1] *= 55
        return final_trajectory



num_points = 6
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
diffusion_outputs = []
#anchors = 32
with torch.no_grad():
    for batch_idx, (instance_feature, map_instance_feature,trajs) in tqdm(enumerate(dataloader)):
        if batch_idx == 100:
            break
        batch_size = instance_feature.shape[0]
        instance_feature,map_instance_feature,trajs = instance_feature.to(device),map_instance_feature.to(device),trajs.to(device)
        #print(trajs.shape)
        trajs_temp =  normalize_xy_rotation(trajs.squeeze(0))
        noise = pyramid_noise_like(trajs_temp)

        diffusion_output = noise

        
        #anchor_size = 128

        #global_cond = instance_feature[:,900:,:]

        #repeated_tensor=global_cond.repeat(1,anchor_size,1)
        #print(repeated_tensor.shape)
        #expanded_tensor=repeated_tensor.view(-1,128)
        #print(expanded_tensor.shape)

        for k in inferece_scheduler.timesteps[:]:
            

            diffusion_output = inferece_scheduler.scale_model_input(diffusion_output)
            noise_pred = model(instance_feature, map_instance_feature,k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),diffusion_output)
            # noise_pred = model.noise_pred_net(
            #         sample=noise,
            #         timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            #         global_cond=global_cond,
            # )
            #print(noise_)
            diffusion_output = inferece_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=diffusion_output
                ).prev_sample
        diffusion_output = denormalize_xy_rotation(diffusion_output, N=num_points, times=10)
        # trajs_output=[]
        #     for i in range(0,batch_size*anchor_size,anchor_size):
        #         path=diffusion_output[i:i+128].mean(0)
        #         trajs_output.append(path)
        #     diffusion_output = torch.stack(trajs_output)
        diffusion_outputs.append(diffusion_output)

# 可以将生成的轨迹保存为文件
with open('generated_trajs_0912_new_test.pkl', 'wb') as f:
    pickle.dump(diffusion_outputs, f)
