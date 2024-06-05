import torch
import torch.nn as nn
from data_loader import TrainDataset
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np
import cv2
from model import Encoder, MFF,CBAMLayer, Decoder, MemoryBank
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
from loss import CosineLoss, Contrast ,SSIMLoss
from test import test,visualization
import torchvision.transforms as transforms
from PIL import Image
import random
import warnings
import re
warnings.filterwarnings('ignore')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

       
def random_mask(x,p):
    mask = torch.bernoulli(torch.full_like(x, 1 - p))
    x = x * mask   
    return x

def train(obj_name, args):
    train_mask = 0.2
    test_mask = 0
    if obj_name in ['road2','road3']:
        train_mask = 0.3
        args.lr = 0.001
    if obj_name in [ 'bottle','cable','capsule','grid','hazelnut','leather', 'metal_nut','pill','screw','tile', 'toothbrush','transistor','wood','zipper','road1']:
        train_mask = 0.2
        test_mask = 0.1
        args.epoch = 200
    resize_shape=256
    print("start train {}".format(obj_name))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # prepare directory
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    
    cur_time = '{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now())
    run_time = str(obj_name) + "_lr" + str(args.lr) + "_bs" + str(args.bs) + "_" + cur_time
    # writer = SummaryWriter(log_dir="./logs/WRes50/"+ run_time +"/")
    os.mkdir("./checkpoints/WRes50/" + run_time)    
    
    # init model
    bank = MemoryBank()
    encoder = Encoder()
    mff = MFF()
    cbam = CBAMLayer()
    decoder = Decoder()
 

    encoder.to(device)
    mff.to(device)
    cbam.to(device)
    decoder.to(device)
    
    encoder.eval()
    
    # init dataloader
    train_dataset = TrainDataset(root_dir=args.data_path, obj_name=obj_name, resize_shape=resize_shape)
    train_loader = DataLoader(
                            train_dataset, 
                            batch_size=args.bs, 
                            shuffle=True,
                            drop_last=True,
                            num_workers=8,
                            persistent_workers=True,
                            pin_memory=True,
                            prefetch_factor=5
                            )
    
    # define loss and optimizer
    mse = nn.MSELoss()
    cos_similarity = CosineLoss()
    contrast = Contrast()

    optimizer = torch.optim.Adam([
      {'params': mff.parameters()},  
      {'params': cbam.parameters()},
      {'params': decoder.parameters()}
], betas=(0.5, 0.999), lr=args.lr)

    #optimizer = torch.optim.Adam(ocbe_decoder.parameters(), betas=(0.5,0.999),lr=args.lr)
    
    auroc_img_best =0
    img_step = 0
    auroc_pix_best = 0
    aupro_px_best = 0
    
    
    # training 
    for step in tqdm(range(args.epochs), ascii=True):
        bank.clear_memory()
        mff.train()
        cbam.train()
        decoder.train()
       
        
        train_loss_total = 0
        index = 0
        for idx, sample in enumerate(train_loader):
            images = sample["image"].to(device)
            
            e_feature1, e_feature2, e_feature3 = encoder(images)
            x = mff(e_feature1, e_feature2, e_feature3)
            x = cbam(x)
            x_split = torch.split(x, 1, dim=0)
            for i in range(len(x_split)):
                bank.add_memory(x_split[i])
            x = random_mask(x , train_mask)
            d_feature1, d_feature2, d_feature3 = decoder(x)
            # loss1 = cos_similarity(e_feature1, d_feature1) + ssim_loss(e_feature1, d_feature1)
            # loss2 = cos_similarity(e_feature2, d_feature2) + ssim_loss(e_feature2, d_feature2)
            # loss3 = cos_similarity(e_feature3, d_feature3) + ssim_loss(e_feature3, d_feature3)
            
            loss1 = cos_similarity(e_feature1, d_feature1)
            loss2 = cos_similarity(e_feature2, d_feature2)
            loss3 = cos_similarity(e_feature3, d_feature3)
            

            
            loss = loss1 + loss2 + loss3
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_total += loss.item()
        
        # writer.add_scalar("train_loss", train_loss_total, int(step))
        if (args.test_interval > 0) and (int(step) % args.test_interval == 0):
            ckp_path = str(args.checkpoint_path + "WRes50/" + run_time  +"/epoch" + str(step) + ".pth")
            torch.save({ 'mff':mff.state_dict(),
                        'cbam':cbam.state_dict(),
                       'decoder':decoder.state_dict()}, ckp_path)
            auroc_img, auroc_pix, aupro_px = test(obj_name=obj_name, ckp_dir=ckp_path, data_dir=args.data_path, reshape_size=resize_shape,memory_bank=bank,test_mask=test_mask)
            # visualization(obj_name=obj_name, ckp_dir="./checkpoints/WRes50/s3/epoch87.pth", 
            #             data_dir=args.data_path, reshape_size=resize_shape,memory_bank=bank)
           
            print('auroc_img:{:.3f},  auroc_pix:{:.3},aupro_px:{:.3f}'.format(auroc_img,  auroc_pix,aupro_px))
            # writer.add_scalar("auroc_img", auroc_img, int(step))
            # writer.add_scalar("auroc_pix", auroc_pix, int(step))
            # writer.add_scalar("test_loss", seg_ap, int(step))
            folder_path = args.checkpoint_path + "WRes50/" + str(run_time) + "/"
            if auroc_img <= auroc_img_best :
                os.remove(ckp_path)
            else  :
                auroc_img_best = auroc_img
                auroc_pix_best = auroc_pix
                aupro_px_best = aupro_px
                img_step = int(step)
                # delete_files(folder_path)
            
    
    return auroc_img_best, auroc_pix_best, aupro_px_best, img_step
            
            
                
            
def write2txt(filename, content):
    f=open(filename,'a')
    f.write(str(content) + "\n")
    f.close()

# def delete_files(folder_path):  
#     # 确保文件夹存在  
#     if not os.path.isdir(folder_path):  
#         print(f"The folder {folder_path} does not exist.")  
#         return  
      
#     # 提取文件名中的数字并找到最大的数字  
#     max_epoch = 0  
#     files = [f for f in os.listdir(folder_path) if f.endswith('.pth')]  
#     for file in files:  
#         # 使用正则表达式提取文件名中的数字  
#         match = re.search(r'epoch(\d+)\.pth', file)  
#         if match:  
#             epoch = int(match.group(1))  
#             max_epoch = max(max_epoch, epoch)  
      
#     # 如果找到了.pth文件，则继续执行删除操作  
#     if max_epoch > 0:  
#         # 构建要保留的文件名模式  
#         keep_file_pattern = f'epoch{max_epoch}.pth'  
          
#         # 删除不匹配的文件  
#         for file in files:  
#             if file != keep_file_pattern:  
#                 file_path = os.path.join(folder_path, file)  
#                 try:  
#                     os.remove(file_path)  
#                 except OSError as e:  
#                     print(f"Error: {e.strerror} : {file_path}")  
              
    
if __name__=="__main__":
    
    setup_seed(11)
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, required=True)
    parser.add_argument('--bs', action='store', type=int, required=False, default=16)
    parser.add_argument('--lr', action='store', type=float, required=False, default=0.005)
    parser.add_argument('--epochs', action='store', type=int, required=False, default=200)
    parser.add_argument('--gpu_id', action='store', type=int, required=False, default=2)
    parser.add_argument('--data_path', action='store', type=str, required=False, default="../datasets/")
    parser.add_argument('--checkpoint_path', action='store', type=str, required=False, default="./checkpoints/")
    # parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--test_interval', action='store',type=int, required=False, default=1)
    
    args = parser.parse_args()
    print(args)
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    obj_names = [
            'road1',
            'road2',
            'road3',
            'tunnel1',
            'tunnel2',
            
            'p01',
            'p02',
            'p03',
            'p04',
           
            'p05',
            'p06',
            'p07',
            'p08',
             
            
             
           
            #  'bottle',
            #  'cable',
            #  'capsule',
            #  'carpet',
            #  'grid',
            #  'hazelnut',
            #  'leather',
            #  'metal_nut',
            #  'pill',
            #  'screw',
            #  'tile',
            #  'toothbrush',
            #  'transistor',
            #  'wood',
            #  'zipper'
             ]
    
    log_txt_name = "./logs_txt/"+str("{date:%Y-%m-%d_%H_%M_%S}".format(date=datetime.datetime.now()))+".txt"
    os.mknod(log_txt_name)
    
    if args.obj_id == -1:
        for obj_name in obj_names:
            auroc_img_best, auroc_pix_best, aupro_px_best,img_step = train(obj_name, args)
            write2txt(log_txt_name, str(obj_name) +" || auroc_img: " + str(auroc_img_best) + " || auroc_pix: " + str(auroc_pix_best)+" || aupro_px:"+str(aupro_px_best)+" epoch:"+str(img_step))
            
    else:
        for obj_id in args.obj_id:
            obj_name = obj_names[obj_id]
            auroc_img_best, auroc_pix_best, aupro_px_best,img_step = train(obj_name, args)
            write2txt(log_txt_name, str(obj_name) + " || auroc_img: " + str(auroc_img_best) + " || auroc_pix: " + str(auroc_pix_best)+" || aupro_px:"+str(aupro_px_best)+" epoch:"+str(img_step))
            
      