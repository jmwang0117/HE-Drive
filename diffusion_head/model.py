from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch
import torch.nn as nn
from conditional_unet1d import ConditionalUnet1D



class CrossAttentionUnetModel(nn.Module):
    def __init__(self, feature_dim, num_heads=8):
        super().__init__()
        # 自车可学习参数
        self.ego_feature = nn.Embedding(1, feature_dim)
        self.feature_dim = feature_dim

        #self.map_feature_pos = self.map_bev_pos.weight[None].repeat(batch_size, 1, 1)


        # 位置编码可学习 MAP / instance 
        

        # Cross-Attention 
        self.ego_instance_decoder = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.ego_map_decoder = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.map_decoder = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        #todo:
        self.fc1 = nn.Sequential(
            nn.Linear(feature_dim, 2 * feature_dim),
            nn.GELU(),
            nn.Linear(2 * feature_dim, feature_dim)  
        )

        self.fc2 = nn.Sequential(
            nn.Linear(feature_dim, 2 * feature_dim),
            nn.GELU(),
            nn.Linear(2* feature_dim, feature_dim) 
        )

        self.ins_cond_layernorm_1 = nn.LayerNorm(self.feature_dim)
        self.ins_cond_layernorm_2 = nn.LayerNorm(self.feature_dim)

        self.map_cond_layernorm_1 = nn.LayerNorm(self.feature_dim)
        self.map_cond_layernorm_2 = nn.LayerNorm(self.feature_dim)

        self.noise_pred_net = ConditionalUnet1D(
                input_dim=20,
                global_cond_dim=256,
                down_dims=[128, 256],
                cond_predict_scale=False,
        )
        self.map_feature_pos = nn.Embedding(100, self.feature_dim)
        self.ego_pos_latent = nn.Embedding(1, self.feature_dim)


        
    def forward(self, instance_feature, map_instance_feature,timesteps,noisy_traj_points):
        batch_size = instance_feature.shape[0]
        #ego_latent = self.ego_feature.weight[None].repeat(batch_size, 1, 1)
        

        map_pos = self.map_feature_pos.weight[None].repeat(batch_size, 1, 1)
        ego_pos = self.ego_pos_latent.weight[None].repeat(batch_size, 1, 1)
        # ego_latent = self.ego_instance_decoder(
        #     query = ego_latent,
        #     key = instance_feature,
        #     value = instance_feature,
        # )[0]
        # ego_latent = self.ins_cond_layernorm_1(ego_latent)
        # #print(ego_latent.shape)
        # ego_latent = self.fc1(ego_latent)
        # ego_latent = self.ins_cond_layernorm_2(ego_latent)

        #ego_latent = ego_latent.unsqueeze(1)
        #print(instance_feature.shape)
        
        ego_latent = instance_feature[:,900:,:]
        #print(ego_latent.shape)
        # map_instance_feature = self.map_decoder(
        #     query = map_instance_feature + map_pos,
        #     key = map_instance_feature + map_pos,
        #     value = map_instance_feature,
        # )[0]
        # map_instance_feature = self.fc1(map_instance_feature)
        # map_instance_feature = self.ins_cond_layernorm_1(map_instance_feature)

        # ego_latent = self.ego_map_decoder(
        #     query = ego_latent + ego_pos,
        #     key = map_instance_feature,
        #     value = map_instance_feature,
        # )[0]

        # ego_latent = self.map_cond_layernorm_1(ego_latent)
        # ego_latent = self.fc2(ego_latent)
        # ego_latent = self.map_cond_layernorm_2(ego_latent)

        global_feature = ego_latent
        global_feature = global_feature.squeeze(1)

        noise_pred = self.noise_pred_net(
                    sample=noisy_traj_points,
                    timestep=timesteps,
                    global_cond=global_feature,
        )
        return noise_pred

if __name__ == "__main__":
    map_feature = torch.randn(2, 100, 256)
    instance_feature = torch.randn(2,901,256)
    global_cond = instance_feature[:,900:,]
    anchor_size = 32
    repeated_tensor=global_cond.repeat(1,anchor_size,1)

    print(repeated_tensor.shape)
    expanded_tensor=repeated_tensor.view(-1,256)
    print(expanded_tensor.shape)
    model = CrossAttentionUnetModel(256)
    noisy_trajs = torch.randn(anchor_size * 2,6,20)

    output = model.noise_pred_net(sample=noisy_trajs, 
                        timestep=torch.tensor([0]),
                        global_cond=expanded_tensor)
    print(output.shape)
    #global_feature = model(instance_feature, map_feature)
    #print(global_feature.shape)
    