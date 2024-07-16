import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import torchvision.models as models
from vit_pytorch import ViT
from transformers import CLIPModel, CLIPProcessor, Wav2Vec2Model
from torch.hub import load
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="dinov2")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation1, dilation2, use_residual=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 320, kernel_size=3, dilation=dilation1, padding=dilation1)
        self.bn1 = nn.BatchNorm1d(320)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv1d(320, 320, kernel_size=3, dilation=dilation2, padding=dilation2)
        self.bn2 = nn.BatchNorm1d(320)
        self.conv3 = nn.Conv1d(320, 320 * 2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(320 * 2)
        self.dropout = nn.Dropout(p=0.3)
        self.glu = nn.GLU(dim=1)
        self.use_residual = use_residual

        if in_channels != 320:
            self.residual = nn.Conv1d(in_channels, 320, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        out = self.gelu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.gelu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        out = self.glu(self.conv3(out))
        out = self.dropout(out)
        return out + residual

class DilatedResidualConvNet(nn.Module):
    def __init__(self, in_channels, depth):
        super(DilatedResidualConvNet, self).__init__()
        layers = []
        for i in range(depth):
            dilation1 = 2 ** (2*i % depth)
            dilation2 = 2 ** ((2*i + 1) % depth)
            use_residual = i != 0
            layers.append(ResidualBlock(in_channels, 320, dilation1, dilation2, use_residual))
            in_channels = 320
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class MEGEncoder(nn.Module):
    def __init__(self, num_subjects, in_channels, out_channels, depth):
        super(MEGEncoder, self).__init__()
        
        self.spatial_attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=1)
        self.subject_layers = nn.ModuleList([nn.Conv1d(in_channels, in_channels, 1, bias=False) for _ in range(num_subjects)])  # 被験者ごとの線形変換
        
        self.dilated_residual_convnet = DilatedResidualConvNet(in_channels, depth)
        self.dropout = nn.Dropout(p=0.2)
        self.self_attention = nn.MultiheadAttention(embed_dim=out_channels * 2, num_heads=8)
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=1, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels *2, kernel_size=1, padding=1)
        self.conv3 = nn.Conv1d(out_channels *2, 2048, kernel_size=1, padding=1)
        self.bn3 = nn.BatchNorm1d(640)
        
        
        self.gelu = nn.GELU()
        self.temporal_projection = nn.Linear(2048, 2048)
        self.temporal_aggregation = nn.Linear(173, 1)

    def forward(self, meg_data, subject_idx): #meg_data:([128, 271, 169])(b, c, t)
        meg_data = meg_data.permute(2, 0, 1)  # (t, b, c)
        meg_data, _ = self.spatial_attention(meg_data, meg_data, meg_data) #torch.Size([169, 128, 271])
        meg_data = meg_data.permute(1, 2, 0)  # (b, c, t)
        meg_data = torch.stack([self.subject_layers[idx.item()](meg_data[i]) for i, idx in enumerate(subject_idx)])  # ([128, 271, 169])
        output = self.dilated_residual_convnet(meg_data) # torch.Size([128, 320, 169])
        output = self.conv2(output) #torch.Size([128, 640, 171])
        output = self.gelu(output) #torch.Size([128, 640, 171])
        output = self.conv3(output) #torch.Size([128, 2048, 173])

        #アフィン射影を使用した時間次元の集約
        output = self.temporal_aggregation(output) #torch.Size([128, 2048, 1])
        output = output.squeeze(-1) #torch.Size([128, 2048])

        # Attentionを使用した時間次元の集約
        #output = output.permute(2, 0, 1)  # 形状を (t, b, c) に変更
        #output, _ = self.self_attention(output, output, output)  # torch.Size([169, 128, 2048])
        #output = output.permute(1, 2, 0)  # (b, c, t) 
        #output = output.mean(dim=2)  # 時間次元を集約

        return output #torch.Size([128, 2048])
    
class MLPHead(nn.Module):
    def __init__(self, input_dim, 
                    hidden_dim1, 
                    hidden_dim2, 
                    output_dim, 
                    dropout_prob=0.2
                ):
        super(MLPHead, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.layer_norm1 = nn.LayerNorm(hidden_dim1)
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_prob)
        
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.layer_norm2 = nn.LayerNorm(hidden_dim2)
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout_prob)
        
        self.layer3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer_norm1(x)
        x = self.gelu1(x)
        x = self.dropout1(x)
        
        x = self.layer2(x)
        x = self.layer_norm2(x)
        x = self.gelu2(x)
        x = self.dropout2(x)
        
        x = self.layer3(x)
        return x

class MEGClassifier(nn.Module):
    def __init__(self, num_classes, num_subjects, in_channels, out_channels, depth, dilation):
        super(MEGClassifier, self).__init__()
        self.meg_encoder = MEGEncoder(num_subjects, in_channels, out_channels, depth)
        self.bn = nn.BatchNorm1d(2560, track_running_stats=True)
        self.dropout = nn.Dropout(p=0.3)
        self.clip_head = MLPHead(2048, hidden_dim1=3072, hidden_dim2=1536, output_dim=512)
        self.mse_head = MLPHead(2048, hidden_dim1=3072, hidden_dim2=1536, output_dim=512)
        self.meg_head = MLPHead(2048, hidden_dim1=3072, hidden_dim2=1536, output_dim=512)
        self.img_head = MLPHead(512, hidden_dim1=1024, hidden_dim2=768, output_dim=512)
        self.output_head = MLPHead(2560, hidden_dim1=3840, hidden_dim2=2880, output_dim=num_classes)

    def forward(self, meg_data, subject_idxs, image_features): # image(torch.Size([128, 512]))
        meg_output = self.meg_encoder(meg_data, subject_idxs) #torch.Size([128, 2048])
        
        combined_output = torch.cat((meg_output, image_features), dim=1)  #torch.Size([128, 2560])
        combined_output = self.bn(combined_output)
        output = self.dropout(combined_output)
        clip_output = self.clip_head(meg_output)
        mse_output = self.mse_head(meg_output)
        img_output = self.img_head(image_features)
        meg_output = self.meg_head(meg_output) #torch.Size([128, 512])
        output = self.output_head(output) #torch.Size([128, 1854])

        #return output, mse_output, clip_output
        return output, meg_output, img_output
    
class ImageEncoder(nn.Module):
    def __init__(self, freeze_layer_indices=[]):
        super(ImageEncoder, self).__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.unfreeze_layers_by_index(freeze_layer_indices)
        self.fc = nn.Linear(512, 768)

    def unfreeze_layers_by_index(self, layer_indices):
        layers = list(self.clip_model.vision_model.encoder.layers)
        for i in layer_indices:
            for param in layers[i].parameters():
                param.requires_grad = True

    def forward(self, images):
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        features = self.clip_model.get_image_features(images)
        features = self.fc(features) 
        return features
