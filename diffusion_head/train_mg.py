import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from model import CrossAttentionUnetModel
from conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch.optim as optim
import torch.nn as nn
filename  = '/home/users/xingyu.zhang/workspace/SparseDrive/scripts/work_dirs/sparsedrive_small_stage2/results.pkl'
features = pickle.load(open(filename, 'rb'))
instance_features = []
map_instance_features = []

device = torch.device("cuda:3")

for i in range(len(features)):
    instance_features.append(features[i]['img_bbox']['instance_feature'])
    map_instance_features.append(features[i]['img_bbox']['map_instance_feature'])
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
        trajs = self.labels[idx]['ego_fut_trajs'].to(device)
        return instance_feature, map_instance_feature,trajs


dataset = FeaturesDataset('/home/users/xingyu.zhang/workspace/SparseDrive/scripts/trajs_all.pkl')

dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

print(dataloader)

model = CrossAttentionUnetModel(feature_dim=256)

noise_pred_net = ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=256,
                down_dims=[128, 256],
                cond_predict_scale=False,
        )

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




optimizer = optim.AdamW(list(model.parameters()) + list(noise_pred_net.parameters()), lr=1e-5, weight_decay=1e-2)

# Define the loss function (MSE Loss for DDPM)
criterion = nn.MSELoss()


num_epochs = 100
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
noise_pred_net.to(device)
for epoch in range(num_epochs):
    model.train()
    noise_pred_net.train()
    for batch_idx, (instance_feature, map_instance_feature,trajs) in enumerate(dataloader):
        #print(batch_idx)
        batch_size = instance_feature.shape[0]
        instance_feature,map_instance_feature,trajs = instance_feature.to(device),map_instance_feature.to(device),trajs.to(device)
        device = instance_feature.device
        noise = torch.randn(batch_size,6,2).to(device)
        timesteps = torch.randint(0, train_scheduler.num_train_timesteps, (batch_size,), dtype=torch.long,device=device)
        noisy_traj_points = train_scheduler.add_noise(
                original_samples=trajs,
                noise=noise,
                timesteps=timesteps,
        ).float()
        global_cond = model(instance_feature, map_instance_feature)
        noise_pred = noise_pred_net(
                    sample=noisy_traj_points,
                    timestep=timesteps,
                    global_cond=global_cond,
        )
        diffusion_loss = criterion(noise_pred, noise)
        optimizer.zero_grad()
        diffusion_loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: { diffusion_loss.item():.4f}")

torch.save({
    'model_state_dict': model.state_dict(),
    'noise_pred_net_state_dict': noise_pred_net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'checkpoint_100epochs.pth')
