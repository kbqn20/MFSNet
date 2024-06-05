import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


def get_ano_map(feature1, feature2):
    mseloss = nn.MSELoss(reduction='none') #1*C*H*W
    mse = mseloss(feature1, feature2) #1*C*H*W
    mse = torch.mean(mse,dim=1) #1*H*W
    cos = nn.functional.cosine_similarity(feature1, feature2, dim=1)
    ano_map = torch.ones_like(cos)-cos
    loss = (ano_map.view(ano_map.shape[0],-1).mean(-1)).mean()
    return ano_map.unsqueeze(1), loss, mse.unsqueeze(1)
    
class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()
        
    def forward(self, feature1, feature2):
        cos = nn.functional.cosine_similarity(feature1, feature2, dim=1)
        ano_map = torch.ones_like(cos) - cos
        loss = (ano_map.view(ano_map.shape[0],-1).mean(-1)).mean()
        return loss
class Contrast(nn.Module):
    def __init__(self):
        super(Contrast, self).__init__()
        
    def forward(self, feature1, feature2):
        feature1 =torch.mean(feature1, axis=0, keepdims=True)
        feature2 =torch.mean(feature2, axis=0, keepdims=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        contrast = torch.nn.CosineEmbeddingLoss(margin = 0.5)
        target = -torch.ones(feature1[0].shape[0]).to(device)
        loss = contrast(feature2.reshape(feature2.shape[0], -1), feature1.reshape(feature1.shape[0], -1), target = target)
        return loss

# x1 = torch.rand(2,10,50,50)

# x2 = torch.rand(2,10,50,50)

# cos = CosineLoss()





def gaussian(window_size, sigma):
    gauss = np.array([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return torch.tensor(gauss / gauss.sum(), dtype=torch.float32)

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True):
    if window is None:
        real_size = min(window_size, img1.size(-1), img1.size(-2))
        window = create_window(real_size, img1.size(1)).to(img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.size(1))
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.size(1)) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean([1, 2, 3])

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
    
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
        
        self.channel = channel
        self.window = window
        
        return 1 - ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)