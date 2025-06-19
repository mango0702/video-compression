import ipdb
from PIL import Image
import datetime
import numpy as np
import torch
import ipdb
import logging
import io
import os
import argparse
#print(torch.cuda.is_available())
from Add.VideoDataset import VideoFolder
from Add.dataset_Copy1 import DataSet
from torch.utils.data import DataLoader
import torch.optim as optim
from src.models.DCVC_DVC_net import DCVC_net
# from src.zoo.image import model_architectures as architectures
from torchvision import transforms
from pytorch_msssim import ms_ssim

lr = 1e-4
# base_lr = 1e-4
decay_rate = 0.1
decay_iter = 10
cal_step=20

root_path = './a_Train_Result/RCVC_u/train_loss'

def save_model(model, epoch):
    if not os.path.exists('./a_Train_Result/RCVC_u/snapshot'): #判断所在目录下是否有该文件名的文件夹
        os.mkdirs('./a_Train_Result/RCVC_u/snapshot')
    torch.save(model.state_dict(), "./a_Train_Result/RCVC_u/snapshot/epoch{}.model".format(epoch))

def imShow(input,i):   
    tensor = transforms.ToPILImage()(input)
#     timestamp = datetime.datetime.now().strftime("%H-%M-%S")#通过时间命名存储结果
#     savepath =  "./Result/Train_1125/img'"
    tensor.save(("./Result/dcvc1/img{}.png".format(i)))
    
def PSNR(input1, input2):
    mse = torch.mean((input1 - input2) ** 2)
    psnr = 20 * torch.log10(1 / torch.sqrt(mse))
    return psnr.item()

def get_updateModel(mv_model, path): 
    pretrained_dict = torch.load(path, map_location='cpu')
    print(len(pretrained_dict))
    mv_model_dict = mv_model.state_dict()
    print(len(mv_model_dict))

    shared_dict = {k: v for k, v in pretrained_dict.items() if k in mv_model_dict}
    print(len(shared_dict))
    
#     for k, v in shared_dict.items():
#         # k 参数名 v 对应参数值        
#         print("model_dict",k)
    #import ipdb; ipdb.set_trace()
    mv_model_dict.update(shared_dict)   
    mv_model.load_state_dict(mv_model_dict, strict=True)
    
#     pretrained_DCVC = torch.load(path_DCVC, map_location='cpu')
#     print(len(pretrained_DCVC))
    for k,v in mv_model.named_parameters():
        if k in pretrained_dict:
             v.requires_grad = False     
    
    return mv_model     
        
def adjust_learning_rate(epoch,optimizer): 
    global lr
    if (epoch % decay_iter):
        lr = lr       
    else:
        lr = lr * decay_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip) 
                
def parse_args():
    parser = argparse.ArgumentParser(description="Example training script")
    
    parser.add_argument('-p', '--pretrain', default='',help='load pretrain model')
    parser.add_argument('-p_DVC', '--pretrain_DVC', default='',help='load pretrain_DVC model')
    parser.add_argument('-p_DCVC', '--pretrain_DCVC', default='',help='load pretrain_DCVC model')
    parser.add_argument('--i_frame_model_name', type=str, default="cheng2020-anchor")
    parser.add_argument('--i_frame_model_path', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument("--num_workers", type=int, default=1, help="worker number")
    parser.add_argument("--patch_size",type=int, default=(256, 256), help="patch size")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--lmbda", type=int, default=256, help="weights")
#     parser.add_argument("--lr", type=float, default=1e-4, help="learning_rate")
    # parser.add_argument("--cuda", type=str2bool, nargs='?', const=True, default=False) 
    args = parser.parse_args()
    return args
        
def train_one_epoch(model_dcvc, train_dataloader, optimizer, epoch, args):
    model_dcvc.train()
    device = next(model_dcvc.parameters()).device
    
    epoch_loss_collector = []
    psnr_collector = []
    ms_ssim_collector = []
    
    for i, batch in enumerate(train_dataloader):
#         ipdb.set_trace()
        d = [frames.to(device) for frames in batch]#2 frames: d[0],d[1]      
#         with torch.no_grad():
#             output_i = model_cheng(d[0])                
#         d[0] = output_i["x_hat"]#d:是len=2的列表，d[0].size=(batch_size,3,256,256) d[0][0].size()=(3,256,256)
    
        output_p = model_dcvc(d[0],d[1],d[2],d[3],d[4],d[5])
        
#         #中间结果可视化
#         imShow(d[0][0],0)#前一重构帧
#         imShow(d[1][0],4)#当前帧       
#         imShow(output_p["warpframe"][0],8)#当前帧的重构帧   
        
        distribution_loss = output_p["bpp"]
        distortion = output_p["mse_loss"]
        distribution_loss_y = output_p["bpp_y"]
        distribution_loss_z = output_p["bpp_z"]
        distribution_loss_mv_y = output_p["bpp_mv_y"]
        distribution_loss_mv_z = output_p["bpp_mv_z"]
        recon_con = output_p["recon_image"]
        recon_res = output_p["recon_image_res"]
        
        distribution_loss_y_res = output_p["bpp_feature_res"]
        distribution_loss_z_res = output_p["bpp_z_res"]
        distribution_loss_res = output_p["bpp_res"]
        distortion_res = output_p["mse_loss_res"]
        
        distribution_loss_all = output_p["bpp_all"]
#         distortion_all = output_p["mse_all"]
#         recon_im_all = output_p["recon_image_all"]
        recon_im_all = 0.5 * recon_con + 0.5 * recon_res
        mse_all = torch.mean((recon_im_all - d[0]).pow(2))
        distortion_all = mse_all
#         rd_loss = args.lmbda * distortion_all + distribution_loss_all + distortion + distortion_res
#         rd_loss = args.lmbda * distortion_all + distribution_loss_all
        rd_loss = distribution_loss_all
    
        optimizer.zero_grad()
        rd_loss.backward()
        clip_gradient(optimizer, 5)
        optimizer.step()
        
        epoch_loss_collector.append(rd_loss.item())
        psnr_collector.append(PSNR(recon_im_all,d[0]))
        ms_ssim_collector.append(ms_ssim(recon_im_all,d[0], data_range=1.0).item())
          
        loss_file_name = '_lr'+str(lr)+'_batch_size'+str(args.batch_size)#+'_lrdecay'+str(lr_decay)     
        loss_file_name1= '_batch_size'+str(args.batch_size)+ '_lr'+str(lr)#+'_lrdecay'+str(lr_decay) 
        
        if (i) % cal_step == 0:          
            with io.open(root_path + loss_file_name + '.txt', 'a', encoding='utf-8') as file: 
                file.write('batch_idx: {:04}\t Loss: {:.6f}\t bpp: {:.6f}\t MSE: {:.6f}\t MSE_con: {:.6f}\t MSE_res: {:.6f}\t psnr: {:.5f}\t ms-ssim: {:.5f}\n'.format(i, epoch_loss_collector[i], distribution_loss_all,args.lmbda * distortion_all, distortion, distortion_res, psnr_collector[i], ms_ssim_collector[i]))
            with io.open(root_path + loss_file_name1 + '.txt', 'a', encoding='utf-8') as file: 
                file.write('batch_idx: {:04}\t bpp: {:.6f}\t y: {:.6f}\t mv: {:.6f}\t y_z: {:.6f}\t mv_z: {:.6f}\t bpp_res: {:.6f}\t y_res: {:.6f}\t y_z_res: {:.6f}\t  \n'.format(i, distribution_loss, distribution_loss_y, distribution_loss_mv_y, distribution_loss_z, distribution_loss_mv_z, distribution_loss_res, distribution_loss_y_res, distribution_loss_z_res))
                
    with io.open(root_path + '.txt', 'a', encoding='utf-8') as file:
        file.write('Train Epoch : {:02} Loss: {:.6f}\t PSNR: {:.5f}\t MS-SSIM: {:.5f}\t lr:{}\n'.format(epoch, sum(epoch_loss_collector) / len(epoch_loss_collector), sum(psnr_collector) / len(psnr_collector),sum(ms_ssim_collector) / len(ms_ssim_collector),lr))
        
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    # if args.seed is not None:
        # torch.manual_seed(args.seed)
        # random.seed(args.seed)
    net_dcvc = DCVC_net()
    if args.pretrain_DVC != '':
        print("loading pretrain : ", args.pretrain_DVC)
        net_dcvc = get_updateModel(net_dcvc, args.pretrain_DVC)
    if args.pretrain_DCVC != '':
        print("loading pretrain : ", args.pretrain_DCVC)
        net_dcvc = get_updateModel(net_dcvc, args.pretrain_DCVC)
    if args.pretrain != '':
        print("loading pretrain : ", args.pretrain)
        net_dcvc = get_updateModel(net_dcvc, args.pretrain)
    net_dcvc = net_dcvc.to(device)
    net_dcvc.train()
    
    net_dcvc_dict = net_dcvc.state_dict()
    for k, v in net_dcvc.named_parameters():      
        # k 参数名 v 对应参数值        
        print("model_dict:",k,"grad:",v.requires_grad)

#     ipdb.set_trace()
    
    train_dataset = DataSet("/mnt/DVC/data/vimeo_septuplet/test_copy.txt")
    train_dataloader = DataLoader(dataset = train_dataset, shuffle=True, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True)
    
    optimizer = optim.Adam(net_dcvc.parameters(), lr)
#     optimizer = optim.Adam(filter(lambda p: p.requires_grad, net_dcvc.parameters()), lr)

    for epoch in range(1,60):
        adjust_learning_rate(epoch,optimizer)
        train_one_epoch(net_dcvc,train_dataloader, optimizer, epoch, args)
        save_model(net_dcvc, epoch)

if __name__ == "__main__":
    main()        