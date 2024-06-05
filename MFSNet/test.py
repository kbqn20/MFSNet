import torch
import torch.nn as nn
from data_loader import TestDataset
from torch.utils.data import DataLoader
import argparse
import os
from model import Encoder, MFF,CBAMLayer,Decoder
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
from torchvision.transforms import ToPILImage
from scipy.ndimage import gaussian_filter
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import average_precision_score
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
from PIL import Image
import random
import pandas as pd
from numpy import ndarray
from skimage import measure
from statistics import mean
import cv2

def get_ano_map(feature1, feature2):
    mseloss = nn.MSELoss(reduction='none') #1*C*H*W
    mse = mseloss(feature1, feature2) #1*C*H*W
    mse = torch.mean(mse,dim=1) #1*H*W
    cos = nn.functional.cosine_similarity(feature1, feature2, dim=1)
    ano_map = torch.ones_like(cos)-cos
    loss = (ano_map.view(ano_map.shape[0],-1).mean(-1)).mean()
    return ano_map.unsqueeze(1), loss, mse.unsqueeze(1)

def mask_feature(f1, f2, mask_ratio=0.05):
    # 计算两个特征图张量之间的欧氏距离
    distances = torch.sqrt(torch.sum((f1 - f2) ** 2, dim=(2, 3)))
    # 计算要屏蔽的特征数量
    num_masked = int(mask_ratio * f1.shape[1])
    # 找到距离最近的特征的索引
    _, indices = torch.topk(distances, num_masked, largest=True) 
    # 创建一个与特征图张量相同形状的掩码张量
    mask = torch.ones_like(f1)
    # 将距离最近的特征值设置为0
    mask[:, indices] = 0
    # 应用掩码
    masked_f1 = f1 * mask
    return masked_f1
    
def test(obj_name, ckp_dir, data_dir, reshape_size, memory_bank,test_mask):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # init model
    encoder = Encoder()
    encoder.to(device)
    mff = MFF()
    mff.to(device)
    cbam = CBAMLayer()
    cbam.to(device)
    decoder = Decoder()
    decoder.to(device)
    
    ckp = torch.load(str(ckp_dir),map_location='cpu')
    mff.load_state_dict(ckp['mff'])
    cbam.load_state_dict(ckp['cbam'])
    decoder.load_state_dict(ckp['decoder'])
    
    encoder.eval()
    mff.eval()
    cbam.eval()
    decoder.eval()
    
    test_dataset = TestDataset(root_dir=data_dir, obj_name=obj_name, resize_shape=reshape_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    test_loss_total = 0
    scores=[]
    labels=[]
    gt_list_px = []
    pr_list_px = []
    aupro_list = []
    with torch.no_grad():
        for idx, sample_test in enumerate(test_loader):
            image, label, gt = sample_test["image"], sample_test["label"], sample_test["gt_mask"]
            
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            
            e_feature1, e_feature2, e_feature3 = encoder(image.to(device))
            x = mff(e_feature1, e_feature2, e_feature3)
            x = cbam(x)
            closest_f = memory_bank.find_closest(x)
            x = mask_feature(x,closest_f,test_mask)
            
            d_feature1, d_feature2, d_feature3 = decoder(x)
            
            ano_map1, loss1, mse1 = get_ano_map(e_feature1, d_feature1)
            ano_map2, loss2, mse2 = get_ano_map(e_feature2, d_feature2)
            ano_map3, loss3, mse3 = get_ano_map(e_feature3, d_feature3)

            # add mse to score
            # ano_map1 = ano_map1 + mse1
            # ano_map2 = ano_map2 + mse2
            # ano_map3 = ano_map3 + mse3
            
            
            ano_map1 = nn.functional.interpolate(ano_map1,size=(reshape_size, reshape_size), mode='bilinear', align_corners=True)
            ano_map2 = nn.functional.interpolate(ano_map2,size=(reshape_size, reshape_size), mode='bilinear', align_corners=True)
            ano_map3 = nn.functional.interpolate(ano_map3,size=(reshape_size, reshape_size), mode='bilinear', align_corners=True)
            s_al = (ano_map1 + ano_map2 + ano_map3).squeeze().cpu().numpy()
            
            s_al = gaussian_filter(s_al, sigma=4)
            
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(s_al.ravel())
            if label.item()!=0:
                aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                              s_al[np.newaxis,:,:]))
            score = np.max(s_al.ravel().tolist())
            
            scores.append(score)
            labels.append(label.numpy().squeeze())
            
            loss = loss1.item() + loss2.item() + loss3.item()
            test_loss_total += loss
            
    auroc_img = round(roc_auc_score(np.array(labels), np.array(scores)), 3)
    auroc_pix = round(roc_auc_score(np.array(gt_list_px), np.array(pr_list_px)), 3)
    # precision, recall, _ = precision_recall_curve(np.array(gt_list_px), np.array(pr_list_px))
    # aupro = auc(recall, precision)
    # seg_ap = average_precision_score(np.array(gt_list_px), np.array(pr_list_px))
    # os.remove(ckp_dir)

    return  auroc_img, auroc_pix, round(np.mean(aupro_list),3)

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# _, auroc_img, auroc_pix= test(obj_name="bottle", ckp_dir="./checkpoints/WRes50/bottle_lr0.001_bs32_2022-03-26_08_01_37/epoch108.pth", data_dir="./datasets/mvtec/" ,reshape_size=256)
# print(auroc_pix)
def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """
    # print(amaps.shape,masks.shape)
    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = pd.concat([df, pd.DataFrame({"pro": mean(pros), "fpr": fpr, "threshold": th}, index=[0])], ignore_index=True)


    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc

def visualization(obj_name, ckp_dir, data_dir, reshape_size, memory_bank):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # init model
    encoder = Encoder()
    encoder.to(device)
    mff = MFF()
    mff.to(device)
    cbam = CBAMLayer()
    cbam.to(device)
    decoder = Decoder()
    decoder.to(device)
    
    ckp = torch.load(str(ckp_dir),map_location='cpu')
    mff.load_state_dict(ckp['mff'])
    cbam.load_state_dict(ckp['cbam'])
    decoder.load_state_dict(ckp['decoder'])
    
    encoder.eval()
    mff.eval()
    cbam.eval()
    decoder.eval()
    
    test_dataset = TestDataset(root_dir=data_dir, obj_name=obj_name, resize_shape=reshape_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    test_loss_total = 0
    scores=[]
    labels=[]
    gt_list_px = []
    pr_list_px = []
    aupro_list = []
    count = 0
    with torch.no_grad():
        for idx, sample_test in enumerate(test_loader):
            image, label, gt = sample_test["image"], sample_test["label"], sample_test["gt_mask"]
            
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            image = image.to(device)
            e_feature1, e_feature2, e_feature3 = encoder(image)
            x = mff(e_feature1, e_feature2, e_feature3)
            x = cbam(x)
            closest_f = memory_bank.find_closest(x)
            x = mask_feature(x,closest_f,0.1)
            d_feature1, d_feature2, d_feature3 = decoder(x)
            
            ano_map1, loss1, mse1 = get_ano_map(e_feature1, d_feature1)
            ano_map2, loss2, mse2 = get_ano_map(e_feature2, d_feature2)
            ano_map3, loss3, mse3 = get_ano_map(e_feature3, d_feature3)

            # add mse to score
            # ano_map1 = ano_map1 + mse1
            # ano_map2 = ano_map2 + mse2
            # ano_map3 = ano_map3 + mse3
            
            
            ano_map1 = nn.functional.interpolate(ano_map1,size=(reshape_size, reshape_size), mode='bilinear', align_corners=True)
            ano_map2 = nn.functional.interpolate(ano_map2,size=(reshape_size, reshape_size), mode='bilinear', align_corners=True)
            ano_map3 = nn.functional.interpolate(ano_map3,size=(reshape_size, reshape_size), mode='bilinear', align_corners=True)
            s_al = (ano_map1 + ano_map2 + ano_map3).squeeze().cpu().numpy()
            s_al = gaussian_filter(s_al, sigma=4)
            ano_map = min_max_norm(s_al)
            ano_map = cvt2heatmap(ano_map*255)
            image = cv2.cvtColor(image.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            image = np.uint8(min_max_norm(image)*255)
            if not os.path.exists('./results_all/'+obj_name):
               os.makedirs('./results_all/'+obj_name)
            cv2.imwrite('./results_all/'+obj_name+'/'+str(count)+'_'+'org.png',image)
           
            ano_map = show_cam_on_image(image, ano_map)
            cv2.imwrite('./results_all/'+obj_name+'/'+str(count)+'_'+'ad.png', ano_map)
            gt = gt.cpu().numpy().astype(int)[0][0]*255
            cv2.imwrite('./results_all/'+obj_name+'/'+str(count)+'_'+'gt.png', gt)
            count += 1
def show_cam_on_image(img, anomaly_map):
    #if anomaly_map.shape != img.shape:
    #    anomaly_map = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap
def get_data_transforms(size, isize):
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        #transforms.CenterCrop(args.input_size),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms