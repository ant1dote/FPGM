import argparse
import logging
import os
import pprint
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import shutil
import yaml, sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from dataset.acdc import KVASIRDataset_H5
from model.unet import UNet_feat
from util.classes import CLASSES
from util.utils import AverageMeter, count_params, init_log, DiceLoss
from util.val_2d import  test_isic_images
import random
from scipy.signal import find_peaks

parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, default='/home/wth/My_codes/SSL_MIS_Exps/Freq_adaptive_modulation/configs/kvasir.yaml')
parser.add_argument('--labeled-id-path', type=str, default='/home/wth/My_codes/SSL_MIS_Exps/polyp/5%_labeled.txt')
parser.add_argument('--unlabeled-id-path', type=str, default='/home/wth/My_codes/SSL_MIS_Exps/polyp/5%_unlabeled.txt')
parser.add_argument('--save-path', type=str, default='/home/wth/My_codes/SSL_MIS_Exps/models/KVASIR')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

args = parser.parse_args()
if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def plot_and_save_profile(profile_tensor, peaks_indices, iteration, save_dir):
    """
    将给定的频率谱和峰值绘制成图表，并保存到指定目录。

    Args:
        profile_tensor (torch.Tensor): 一维的频率谱张量。
        peaks_indices (list or np.array): 检测到的峰值索引。
        iteration (int): 当前的训练迭代次数。
        save_dir (str): 保存图像的目录。
    """
    if profile_tensor is None:
        return

    profile_np = profile_tensor.cpu().numpy()
    
    # --- 绘图设置 ---
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6), dpi=1000)
    plt.rcParams['font.family'] = 'Arial'

    # --- 绘制主曲线和峰值 ---
    plt.plot(profile_np, label='Running-Mean Frequency Profile (P)', color='dodgerblue', lw=2.5)
    
    if len(peaks_indices) > 0:
        peak_values = profile_np[peaks_indices]
        plt.scatter(
            peaks_indices, 
            peak_values, 
            color='red', 
            marker='x', 
            s=100,
            lw=2.5,
            label='Detected Peaks for Suppression (P*)',
            zorder=5
        )
    
    # --- 美化图表 ---
    plt.title(f'Learned Frequency Prior at Iteration: {iteration}', fontsize=16)
    plt.xlabel('Frequency Radius', fontsize=12)
    plt.ylabel('Mean Amplitude', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlim(0, 16)
    plt.ylim(bottom=0) # 确保Y轴从0开始

    # --- 保存图像 ---
    # 创建一个专门的子目录来存放这些演化图
    evolution_dir = os.path.join(save_dir, 'freq_profile_evolution')
    os.makedirs(evolution_dir, exist_ok=True)
    
    output_path = os.path.join(evolution_dir, f'iter_{iteration}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # 关闭图形，释放内存，这在长时训练中非常重要！
    plt.close()

class FreqPerturbation(nn.Module):
    """
    通过频谱形状匹配 (Spectral Shape Matching) 进行图像扰动。

    此模块通过以下步骤实现理论上更严谨的知识迁移：
    1. 从有标签数据的“纯净”边缘信号中学习频率先验 P_prior 的“形状”。
    2. 对无标签的“混杂”完整图像频谱 P_u 进行归一化，提取其“形状”。
    3. 在形状空间进行插值，生成新的频谱形状 P_pert_norm。
    4. 将新的形状乘以 P_u 的原始能量，恢复其尺度，完成扰动。
    """
    def __init__(self, 
                 gamma: float = 0.05, 
                 momentum: float = 0.999, 
                 dilation_kernel_size: int = 3,
                 eps: float = 1e-6): # 增加一个小的 epsilon 防止除以零
        """
        初始化。

        Args:
            gamma (float): 形状迁移强度/插值系数。
            momentum (float): EMA 更新系数。
            dilation_kernel_size (int): 边缘扩张核大小。
            eps (float): 用于归一化时防止除以零的小常数。
        """
        super().__init__()
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("gamma must be between 0 and 1.")
            
        self.gamma = gamma
        self.momentum = momentum
        self.dilation_kernel_size = dilation_kernel_size
        self.eps = eps
        
        # --- 辅助模块 (保持不变) ---
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        rgb_to_gray_weights = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer('rgb_to_gray_weights', rgb_to_gray_weights)
        
        self.register_buffer('running_mean_freq_profile', None)

    # =======================================================================
    #       【学习先验的函数 - 保持不变】
    # =======================================================================
    @torch.no_grad()
    def _get_edge_mask(self, mask: torch.Tensor) -> torch.Tensor:
        # (代码与您提供的版本相同，但确保设备对齐)
        device = mask.device
        mask_float = mask.unsqueeze(1).float()
        edge_x = F.conv2d(mask_float, self.sobel_x.to(device), padding=1)
        edge_y = F.conv2d(mask_float, self.sobel_y.to(device), padding=1)
        edge = torch.sqrt(edge_x.pow(2) + edge_y.pow(2))
        dilated_edge = F.max_pool2d(edge, kernel_size=self.dilation_kernel_size, stride=1, padding=self.dilation_kernel_size // 2)
        return (dilated_edge > 0.5)

    @torch.no_grad()
    def _get_radial_profile(self, fft_map_amp: torch.Tensor) -> torch.Tensor:
        # (使用最鲁棒的循环版本)
        B, H, W = fft_map_amp.shape
        center_h, center_w = H // 2, W // 2
        y, x = torch.meshgrid(torch.arange(H, device=fft_map_amp.device), torch.arange(W, device=fft_map_amp.device), indexing='ij')
        radius = torch.sqrt((y - center_h).float().pow(2) + (x - center_w).float().pow(2)).long()
        max_radius = min(center_h, center_w)
        all_profiles = torch.zeros(B, max_radius, device=fft_map_amp.device)
        for b in range(B):
            for r in range(max_radius):
                mask = (radius == r)
                if mask.any():
                    all_profiles[b, r] = fft_map_amp[b, mask].mean()
        return all_profiles
    
    @torch.no_grad()
    def update_freq_prior(self, image: torch.Tensor, gt_mask: torch.Tensor):
        # (学习先验的逻辑保持不变，依然从边缘学习)
        B, C, H, W = image.shape
        device = image.device
        edge_mask = self._get_edge_mask(gt_mask)
        if C == 3:
            image_gray = torch.sum(image * self.rgb_to_gray_weights.to(device), dim=1, keepdim=True)
        else:
            image_gray = image
        
        image_at_edge = image_gray * edge_mask
        fft_complex = torch.fft.fft2(image_at_edge, norm='ortho')
        fft_amp = torch.abs(fft_complex)
        fft_amp_shifted = torch.fft.fftshift(fft_amp, dim=(-2, -1))
        
        current_profile = self._get_radial_profile(fft_amp_shifted.squeeze(1)).mean(dim=0)
        
        if self.running_mean_freq_profile is None:
            self.running_mean_freq_profile = current_profile
        else:
            min_len = min(self.running_mean_freq_profile.shape[0], current_profile.shape[0])
            self.running_mean_freq_profile = self.running_mean_freq_profile[:min_len]
            current_profile = current_profile[:min_len]
            self.running_mean_freq_profile.mul_(self.momentum).add_(current_profile, alpha=1 - self.momentum)

    # =======================================================================
    #          【核心修改：实现了形状匹配的 FORWARD 方法】
    # =======================================================================
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """对输入图像施加频谱形状匹配扰动。"""
        if self.running_mean_freq_profile is None or self.gamma == 0:
            return image
    
        B, C, H, W = image.shape
        device = image.device
        P_prior = self.running_mean_freq_profile
    
        # --- 1. 傅里叶变换，分离振幅和相位 ---
        fft_complex = torch.fft.fft2(image, dim=(-2, -1), norm='ortho')
        fft_amp = torch.abs(fft_complex)
        fft_phase = torch.angle(fft_complex)
    
        # --- 2. 计算 P_u 并提取其形状和能量 ---
        fft_amp_shifted = torch.fft.fftshift(fft_amp, dim=(-2, -1))
        # P_u_channels 的形状是 [B, C, max_radius]
        P_u_channels = self._get_radial_profile(fft_amp_shifted.view(-1, H, W)).view(B, C, -1)
        
        # 计算每个通道的原始能量 (L1范数)
        P_u_energy = torch.sum(P_u_channels, dim=2, keepdim=True)
        # 归一化，得到 P_u 的形状
        P_u_norm = P_u_channels / (P_u_energy + self.eps)
        
        # --- 3. 归一化 P_prior 并进行形状插值 ---
        effective_radius = min(P_prior.shape[0], P_u_channels.shape[2])
        
        P_prior_c = P_prior[:effective_radius]
        # 归一化，得到 P_prior 的形状
        P_prior_norm = P_prior_c / (torch.sum(P_prior_c) + self.eps)
        # 扩展以进行广播运算
        P_prior_norm = P_prior_norm.view(1, 1, -1).expand(B, C, -1)
        
        # 在形状空间进行插值
        P_pert_norm = (1 - self.gamma) * P_u_norm[:, :, :effective_radius] + self.gamma * P_prior_norm
        
        # --- 4. 恢复能量尺度 ---
        P_perturbed = P_pert_norm * P_u_energy[:, :, :1] # 乘以原始能量
        
        # --- 5. 广播回 2D 振幅图 ---
        center_h, center_w = H // 2, W // 2
        y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        radius = torch.sqrt((y - center_h).float().pow(2) + (x - center_w).float().pow(2)).long()
        clamped_radius = torch.clamp(radius, max=effective_radius - 1)
        
        expanded_radius = clamped_radius.expand(B, C, H, W)
        Amp_perturbed = torch.gather(P_perturbed.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, H, W), 2, expanded_radius.unsqueeze(2)).squeeze(2)

        # --- 6. 逆变换，重构图像 ---
        Amp_perturbed_unshifted = torch.fft.ifftshift(Amp_perturbed, dim=(-2, -1))
        fft_complex_perturbed = Amp_perturbed_unshifted.to(torch.complex64) * torch.exp(1j * fft_phase)
        perturbed_image = torch.fft.ifft2(fft_complex_perturbed, dim=(-2, -1), norm='ortho').real
        
        return perturbed_image

def main(snapshot_path):
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = 0, 1

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    model = UNet_feat(in_chns=3, class_num=cfg['nclass'])    
    ema_model = UNet_feat(in_chns=3, class_num=cfg['nclass'])
    for parameter in ema_model.parameters():
        parameter.detach_()
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))
        
    optimizer = SGD(model.parameters(), cfg['lr'], momentum=0.9, weight_decay=0.0001)
    
    model.cuda()
    ema_model.cuda()

    freq_perturber = FreqPerturbation(gamma=0.05)#HeuristicFilterPerturbation('low-pass', 16)#RandomBandPerturbation(alpha=0.15, num_peaks=3)
    freq_perturber.cuda()

    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss(n_classes=cfg['nclass'])
    
    trainset_u = KVASIRDataset_H5(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = KVASIRDataset_H5(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = KVASIRDataset_H5(cfg['dataset'], cfg['data_root'], 'val')

    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                                pin_memory=True, num_workers=1, drop_last=True)
    
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                                pin_memory=True, num_workers=1, drop_last=True)
    
    trainloader_u_mix = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                                pin_memory=True, num_workers=1, drop_last=True)
    
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False)
    
    previous_best = 0.0
    epoch = -1
   
    iter_num = 0
    max_iterations = 30000
    max_epoch = 100000#max_iterations // (cfg['batch_size']-1) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    num_classes = cfg['nclass']

    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_mask_ratio = AverageMeter()
        
        loader = zip(trainloader_l, trainloader_u, trainloader_u_mix)

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, cutmix_box1, cutmix_box2),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, _, _)) in enumerate(loader):
            
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2 = img_u_s1.cuda(), img_u_s2.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()

            freq_perturber.train()
            freq_perturber.update_freq_prior(img_x, mask_x)

            with torch.no_grad():
                model.eval()
                pred_u_w_mix, feat_u_w_mix = model(img_u_w_mix)
                # 计算原始预测的硬标签和置信度
                prob_u_w_mix = pred_u_w_mix.softmax(dim=1)
                conf_u_w_mix, mask_u_w_mix = prob_u_w_mix.max(dim=1)

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
            
            model.train()

            # 4. 学生模型前向传播
            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            # 有标签 和 无标签弱增强 (带特征扰动)
            (preds, preds_fp), (feat, _) = model(torch.cat((img_x, img_u_w)), True)
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            feat_x, feat_u_w = feat.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]

            # 强增强
            pred_s1, _ = model(img_u_s1)
            pred_s2, _ = model(img_u_s2)
            pred_u_s = torch.cat((pred_s1, pred_s2), dim=0)
            
            with torch.no_grad():

                # 计算原始预测的硬标签和置信度
                prob_u_w = pred_u_w.softmax(dim=1)
                conf_u_w, mask_u_w = prob_u_w.max(dim=1)

            with torch.no_grad():
                mask_teacher1 = mask_u_w.clone()
                conf_teacher1 = conf_u_w.clone()

                # 分支2 (对应 img_u_s2)
                mask_teacher2 = mask_u_w.clone()
                conf_teacher2 = conf_u_w.clone()

                # --- 对分支1进行混合 ---
                mask_teacher1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
                conf_teacher1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]

                                                            
                # --- 对分支2进行混合 ---
                mask_teacher2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
                conf_teacher2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]

                # --- 将两个分支的教师信号合并 ---
                # 硬伪标签 (用于HC-Loss)
                mask_teacher_final = torch.cat((mask_teacher1, mask_teacher2), dim=0)
                # 置信度 (用于定义区域)
                conf_teacher_final = torch.cat((conf_teacher1, conf_teacher2), dim=0)
                # 软伪标签 (用于MC-Loss)
               
            '''
            if random.random() <= 0.7:
                img_u_w_freq = freq_perturber(img_u_w.detach())
                pred_u_w_freq, _ = model(img_u_w_freq)
                loss_freq1 =criterion_dice(pred_u_w_freq.softmax(dim=1), mask_u_w.unsqueeze(1).float(), ignore=(conf_u_w < cfg['conf_thresh_high']))
            else:
                loss_freq1 = 0.0

            if random.random() <= 0.7:
                img_u_w_mix_freq = freq_perturber(img_u_w_mix.detach())
                pred_u_w_mix_freq, _ = model(img_u_w_mix_freq)
                loss_freq2 = criterion_dice(pred_u_w_mix_freq.softmax(dim=1), mask_u_w_mix.unsqueeze(1).float(), ignore=(conf_u_w_mix < cfg['conf_thresh_high']))
            else:
                loss_freq2 = 0.0
            

            loss_freq = (loss_freq1 + loss_freq2) / 2.0
            '''

            img_u_w_freq = freq_perturber(img_u_w.detach())
            pred_u_w_freq, _ = model(img_u_w_freq)
            loss_freq = criterion_dice(pred_u_w_freq.softmax(dim=1), mask_u_w.unsqueeze(1).float(), ignore=(conf_u_w < cfg['conf_thresh_high']))

            loss_hc = criterion_dice(pred_u_s.softmax(dim=1), 
                             mask_teacher_final.unsqueeze(1).float(),
                             ignore=(conf_teacher_final < cfg['conf_thresh_high']))

            loss_x = (criterion_ce(pred_x, mask_x) + criterion_dice(pred_x.softmax(dim=1), mask_x.unsqueeze(1).float())) / 2.0
            
            # 7.4 特征扰动损失 (FP-Loss)
            loss_fp = criterion_dice(pred_u_w_fp.softmax(dim=1), mask_u_w.unsqueeze(1).float(),
                             ignore=(conf_u_w < cfg['conf_thresh_high']))
            
            loss = loss_x + loss_hc * 0.5 + loss_fp * 0.5 + loss_freq * 0.5 
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_hc.item())
            total_loss_w_fp.update(loss_fp.item())
            
            mask_ratio = (conf_u_w >= cfg['conf_thresh']).sum() / conf_u_w.numel()
            total_mask_ratio.update(mask_ratio.item())
            
            lr = cfg['lr'] * (1 - iter_num / max_iterations) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            
            iter_num += 1   

            logging.info('iteration %d: loss: %f, loss_s1: %f, loss_fp: %f, loss_freq: %f'%(iter_num, loss, loss_hc, loss_fp, loss_freq))
                
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, (img, mask) in enumerate(valloader):
                    metric_i = test_isic_images(img, mask, model, classes=2)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(valset)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format('unet'))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()
            
            #if iter_num == 1 or (iter_num > 0 and iter_num % 200 == 0):
                
                # 获取当前的全局平均谱
            '''
            current_profile = freq_perturber.running_mean_freq_profile
                
            if current_profile is not None:
                    # 在这个谱线上寻找峰值
                detected_peaks = freq_perturber._find_freq_peaks(current_profile.detach())
                    
                # 调用绘图函数
                    # snapshot_path 是您实验的保存目录
                if len(detected_peaks)>0:
                    plot_and_save_profile(current_profile.detach(), 
                                          detected_peaks, 
                                          iter_num, 
                                          snapshot_path)
            '''
            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()

if __name__ == '__main__':
    snapshot_path = "/home/wth/My_codes/SSL_MIS_Exps/Freq_adaptive_modulation/results/polyp_{}_{}labeled/".format('unimatch_freq_prior_matchingnorm', 70)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    shutil.copy('/home/wth/My_codes/SSL_MIS_Exps/Freq_adaptive_modulation/poly_freq_prior.py', snapshot_path)
    main(snapshot_path)
   
