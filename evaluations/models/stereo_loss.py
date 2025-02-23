import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('..')
from config import params
from models import *
from utils import utils, torch_utils

from sklearn.metrics import average_precision_score
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
# ----------------------------------------------------------------------- # 

class BCELoss(nn.Module):
    # binary classification loss
    def __init__(self, args, pr, device):
        super(BCELoss, self).__init__()
        self.pr = pr
        # self.class_dist = pr.class_dist
        self.class_info = pr.class_info
        self.device = device
        self.pos_weight = torch.tensor(pr.data_weight).to(device)
        self.criterion = nn.BCELoss(reduction='mean')

    def forward(self, pred, target):
        # import pdb; pdb.set_trace()
        loss = self.criterion(pred, target.float())
        return loss

    def evaluate(self, pred, target):
        loss = self.criterion(pred, target.float())
        # pred = torch.sigmoid(pred)
        pred = pred.data.cpu().numpy()
        target = target.data.cpu().numpy()
        ap = average_precision_score(target, pred)
        acc = torch_utils.binary_acc(pred, target, thred=0.5)
        res = {
            'Loss': loss.item(),
            'AP': ap,
            'Acc': acc
        }
        return res

    # def _generate_class_weight(self):
    #     # import pdb; pdb.set_trace()
    #     class_dist = torch.from_numpy(self.class_dist).float()
    #     pos_weights = torch.tensor([class_dist[0] / class_dist[1]])
    #     return pos_weights


class BCEwithLogitsLoss(BCELoss):
    # binary classification loss
    def __init__(self, args, pr, device):
        super(BCEwithLogitsLoss, self).__init__(args, pr, device)

        self.criterion = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=self.pos_weight)
    
    def evaluate(self, pred, target):
        loss = self.criterion(pred, target.float())
        pred = torch.sigmoid(pred)
        pred = pred.data.cpu().numpy()
        target = target.data.cpu().numpy()
        ap = average_precision_score(target, pred)
        acc = torch_utils.binary_acc(pred, target, thred=0.5)
        res = {
            'Loss': loss.item(),
            'AP': ap,
            'Acc': acc
        }
        return res
# ----------------------------------------------------------------------- # 

class MCLoss(nn.Module):
    # multi-classification loss
    def __init__(self, args, pr, device):
        super(MCLoss, self).__init__()
        self.pr = pr
        # self.class_dist = pr.class_dist
        # self.class_info = pr.class_info
        self.device = device
        self.class_weight = torch.tensor(pr.data_weight).to(device)
        self.criterion = nn.CrossEntropyLoss(reduction='mean', weight=self.class_weight.float())
    
    def forward(self, pred, target):
        # import pdb; pdb.set_trace()
        loss = self.criterion(pred, target.long())
        return loss

    def evaluate(self, pred, target):
        loss = self.criterion(pred, target.long())
        pred = F.softmax(pred, dim=-1)
        top_1_acc = torch_utils.calc_acc(pred, target, k=1)
        top_5_acc = torch_utils.calc_acc(pred, target, k=5)
        micro_f1 = self.calc_f1(pred, target)
        avg_distance = self.calc_avgdis(pred, target)
        res =  {
            'loss': loss.item(),
            'Top-1 acc': top_1_acc.item(),
            'Top-5 acc': top_5_acc.item(),
            'micro_f1': micro_f1,
            'Avg Dis.': avg_distance
        }
        return res
    
    def calc_f1(self, pred, target):
        pred = torch.argmax(pred, dim=-1) 
        pred = pred.data.cpu().numpy()
        target = target.data.cpu().numpy()
        f1 = f1_score(target, pred, average='micro')
        return f1

    def calc_avgdis(self, pred, target, unit=1):
        pred = torch.argmax(pred, dim=-1) 
        pred = pred.data.cpu().numpy()
        target = target.data.cpu().numpy()
        avg_dis = np.abs(pred - target) * unit
        avg_dis = np.mean(avg_dis)
        return avg_dis




# ----------------------------------------------------------------------- # 


class ARLoss(nn.Module):
    # audio regression loss
    def __init__(self, args, pr, device):
        super(ARLoss, self).__init__()
        self.pr = pr
        self.device = device
        # pos_weight = self._generate_class_weight().to(device)
        # if args.loss == 'MSE':
        self.criterion = nn.MSELoss(reduction='mean')
        # elif args.loss == 'MAE':
        #     self.criterion = nn.L1Loss(reduction='mean')
        # elif args.loss == 'Huber':
        #     self.criterion = HuberLoss(delta=args.delta)

    def forward(self, pred, target):
        import pdb; pdb.set_trace()
        loss = self.criterion(pred, target.float())
        return loss

    def evaluate(self, pred, target):
        pred = pred.float()
        target = target.float()
        loss = self.criterion(pred, target)
        errs = torch.abs(pred - target)
        mean_err = torch.mean(errs)
        median_err = torch.median(errs)
        pred = pred.data.cpu().numpy()
        target = target.data.cpu().numpy()
        r2 = r2_score(target, pred)
        res = {
            'Loss': loss.view(1, -1),
            'Mean Error': mean_err.view(1, -1),
            'Median Error': median_err.view(1, -1),
            'R2 Score': torch.tensor(r2).view(1, -1).to(loss.device)
        }
        return res


class HuberLoss(torch.nn.Module):
    def __init__(self, delta):
        super(HuberLoss, self).__init__()
        self.delta = delta
        
    def forward(self, y_predict, y):
        residual = torch.abs(torch.sub(y, y_predict))
        mask1 = residual <= self.delta
        lossMSE = torch.sum(0.5 * torch.pow(residual[mask1], 2))
        mask2 = residual > self.delta
        lossMAE = torch.sum(torch.sub(self.delta * residual[mask2], 0.5 * self.delta**2))
        
        return (lossMSE + lossMAE) / (y.size()[0])

# ----------------------------------------------------------------------- # 

class StereoCRWLoss(nn.Module):
    # audio-visual co learning loss
    def __init__(self, args, pr, device):
        super(StereoCRWLoss, self).__init__()
        self.args = args
        self.pr = pr
        self.device = device
        self.tau = pr.tau
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.smooth_criterion = nn.SmoothL1Loss(beta=1.0)
        self.eps = 1e-20
        self.skip_node = args.skip_node
        self.smooth = args.smooth
        self.cycle_num = args.cycle_num
        self.bidirectional = args.bidirectional

    def forward(self, feat_dict, inference=False):
        # import pdb; pdb.set_trace()
        feat_left = feat_dict['audio_left'] 
        feat_left = F.normalize(feat_left, p=2, dim=-1)
        feat_right = feat_dict['audio_right'] 
        feat_right = F.normalize(feat_right, p=2, dim=-1)

        aff_L2R = self.feat2mat(feat_left, feat_right)
        aff_R2L = self.feat2mat(feat_right, feat_left)

        cycle_matrix, labels = self.calc_cycle_matrix(aff_L2R, aff_R2L)
        if self.bidirectional:
            cycle_matrix_R, labels_R = self.calc_cycle_matrix(aff_R2L, aff_L2R)
            cycle_matrix = torch.cat([cycle_matrix, cycle_matrix_R], dim=1)
            labels = torch.cat([labels, labels_R], dim=1)

        cycle_matrix = cycle_matrix.contiguous().view(-1, cycle_matrix.shape[-1])
        labels = labels.contiguous().view(-1)
        crw_loss = self.criterion(torch.log(cycle_matrix + self.eps), labels.long())
        smooth_loss = self.calc_smooth_loss(aff_L2R, feat_dict['delay_time'])
        loss = crw_loss + smooth_loss
        if inference:
            return {
                'pred': cycle_matrix,
                'target': labels,
                'crw loss': crw_loss,
                'smooth loss': smooth_loss,
                'loss': loss
            }
        return loss


    def feat2mat(self, feat_1, feat_2):
        # import pdb; pdb.set_trace()
        aff = torch.matmul(feat_1, feat_2.transpose(-2, -1)) / self.tau
        return aff

    def calc_cycle_matrix(self, aff_1, aff_2):
        # import pdb; pdb.set_trace()
        aff_1 = F.softmax(aff_1, dim=-1)
        aff_2 = F.softmax(aff_2, dim=-1)
        cycle_matrix = torch.matmul(aff_1, aff_2)
        cycle_matrix = torch.matrix_power(cycle_matrix, self.cycle_num)
        # skip = self.skip_node
        N, M, _ = cycle_matrix.shape
        target = torch.eye(M).repeat(N, 1, 1).to(cycle_matrix.device)
        _, labels = target.max(dim=-1)
        if self.skip_node:
            skip = np.floor(self.pr.samp_sr * self.pr.max_delay / self.pr.patch_stride).astype(int)
            cycle_matrix = cycle_matrix[:, skip:-skip, :]
            labels = labels[:, skip:-skip]
        return cycle_matrix, labels

    def evaluate(self, feat_dict):
        out = self.forward(feat_dict, inference=True)
        res = self.evaluate_metrics(out['pred'], out['target'])
        res['loss'] = out['loss'].view(1, -1)
        
        if self.smooth > 0:
            res['smooth loss'] = out['smooth loss'].view(1, -1)
            res['crw loss'] = out['crw loss'].view(1, -1)

        return res
    
    def evaluate_metrics(self, pred, target):
        # import pdb; pdb.set_trace()
        # loss = self.criterion(torch.log(pred + self.eps), target.long())
        top_1_acc = torch_utils.calc_acc(pred, target, k=1)
        top_3_acc = torch_utils.calc_acc(pred, target, k=3)
        res =  {
            'Top-1 acc': top_1_acc.view(1, -1),
            'Top-3 acc': top_3_acc.view(1, -1)
        }
        return res

    def inference(self, feat_dict, softmax=True):
        # import pdb; pdb.set_trace()
        feat_left = feat_dict['audio_left'] 
        feat_left = F.normalize(feat_left, p=2, dim=-1)
        feat_right = feat_dict['audio_right'] 
        feat_right = F.normalize(feat_right, p=2, dim=-1)

        aff_L2R = self.feat2mat(feat_left, feat_right)
        if softmax:
            aff_L2R = F.softmax(aff_L2R, dim=-1)

        return aff_L2R
    
    def calc_smooth_loss(self, aff_L2R, delay_time):
        # import pdb; pdb.set_trace()
        if self.smooth <= 0:
            return torch.tensor(0.0).to(delay_time.device)
        crw_itd = self.crw_itd_estimation(aff_L2R, delay_time)
        itd_diff = crw_itd[:, :-1] - crw_itd[:, 1:]
        # itd_diff = itd_diff * 1e2
        target = torch.zeros_like(itd_diff).float()
        loss = self.smooth_criterion(itd_diff, target)
        loss = loss * self.smooth
        return loss

    def crw_itd_estimation(self, aff_L2R, delay_time):
        # import pdb; pdb.set_trace()
        max_delay_offset = np.floor(self.pr.samp_sr * self.pr.max_delay / self.pr.patch_stride).astype(int)
        N, H, W = aff_L2R.shape
        repeats = torch.eye(H, W).view(-1).long() * (max_delay_offset * 2) + 1
        mask = torch.eye(H, W).view(-1).repeat_interleave(repeats, dim=-1).view(H, -1)
        mask = mask.repeat(N, 1, 1).to(aff_L2R.device)
        mask = mask[:, :, max_delay_offset : -max_delay_offset]
        masked_aff = aff_L2R * mask
        masked_aff[masked_aff == 0] = -1e20
        masked_aff = F.softmax(masked_aff, dim=-1)
        delay_time = delay_time * self.pr.samp_sr
        crw_itd = torch.sum(masked_aff * delay_time, dim=-1)
        crw_itd = crw_itd[:, max_delay_offset: -max_delay_offset]
        return crw_itd


class StereoNCELoss(StereoCRWLoss):
    # InfoNCE baseline learning loss
    def __init__(self, args, pr, device):
        super(StereoNCELoss, self).__init__(args, pr, device)

    
    def forward(self, feat_dict, inference=False):
        # import pdb; pdb.set_trace()
        feat_left = feat_dict['audio_left'] 
        feat_left = F.normalize(feat_left, p=2, dim=-1)
        feat_right = feat_dict['audio_right'] 
        feat_right = F.normalize(feat_right, p=2, dim=-1)

        aff_L2R = self.feat2mat(feat_left, feat_right)
        aff_R2L = self.feat2mat(feat_right, feat_left)

        N, M, _ = aff_L2R.shape
        target = torch.eye(M).repeat(N, 1, 1).to(aff_L2R.device)
        _, labels = target.max(dim=-1)
        affinity_matrix = aff_L2R
        if self.bidirectional:
            affinity_matrix = torch.cat([affinity_matrix, aff_R2L], dim=1)
            labels = torch.cat([labels, labels], dim=1)

        affinity_matrix = affinity_matrix.contiguous().view(-1, affinity_matrix.shape[-1])
        labels = labels.contiguous().view(-1)
        nce_loss = self.criterion(affinity_matrix, labels.long())
        smooth_loss = self.calc_smooth_loss(aff_L2R, feat_dict['delay_time'])
        loss = nce_loss + smooth_loss
        if inference:
            return {
                'pred': affinity_matrix,
                'target': labels,
                'crw loss': nce_loss,
                'smooth loss': smooth_loss,
                'loss': loss
            }
        return loss


class StereoCRWAugLoss(StereoCRWLoss):
    # audio-visual co learning loss
    def __init__(self, args, pr, device):
        super(StereoCRWAugLoss, self).__init__(args, pr, device)
        self.synthetic_rate = args.synthetic_rate
        self.crw_rate = args.crw_rate

    
    def forward(self, feat_dict, inference=False):
        # import pdb; pdb.set_trace()
        feat_left = feat_dict['audio_left'] 
        feat_left = F.normalize(feat_left, p=2, dim=-1)
        feat_right = feat_dict['audio_right'] 
        feat_right = F.normalize(feat_right, p=2, dim=-1)
        feat_noaugleft = feat_dict['audio_noaug_left'] 
        feat_noaugleft = F.normalize(feat_noaugleft, p=2, dim=-1)

        aff_L2R = self.feat2mat(feat_left, feat_right)
        aff_R2L = self.feat2mat(feat_right, feat_noaugleft)
        aff_L2SL = self.feat2mat(feat_left, feat_noaugleft)

        cycle_matrix, crw_labels = self.calc_cycle_matrix(aff_L2R, aff_R2L, feat_dict['shift_offset'])
        if self.bidirectional:
            cycle_matrix_R, labels_R = self.calc_cycle_matrix(aff_R2L, aff_L2R, feat_dict['shift_offset'])
            cycle_matrix = torch.cat([cycle_matrix, cycle_matrix_R], dim=1)
            crw_labels = torch.cat([crw_labels, labels_R], dim=1)
        cycle_matrix = cycle_matrix.contiguous().view(-1, cycle_matrix.shape[-1])
        crw_labels = crw_labels.contiguous().view(-1)
        crw_loss = self.criterion(torch.log(cycle_matrix + self.eps), crw_labels.long())

        syn_aff, aff_labels = self.calculate_syn_matrix(aff_L2SL, feat_dict['shift_offset'])
        syn_loss = self.criterion(syn_aff, aff_labels.long())

        smooth_loss = self.calc_smooth_loss(aff_L2R, feat_dict['delay_time'])

        loss = self.crw_rate * crw_loss + self.synthetic_rate * syn_loss + smooth_loss
        if inference:
            return {
                'pred_crw': cycle_matrix,
                'target_crw': crw_labels,
                'pred_syn': syn_aff,
                'target_syn': aff_labels,
                'loss': loss,
                'crw loss': crw_loss,
                'smooth loss': smooth_loss,
                'syn loss': syn_loss
            }
        return loss

    def calc_cycle_matrix(self, aff_1, aff_2, shift_offset):
        # import pdb; pdb.set_trace()
        aff_1 = F.softmax(aff_1, dim=-1)
        aff_2 = F.softmax(aff_2, dim=-1)
        cycle_matrix = torch.matmul(aff_1, aff_2)
        cycle_matrix = torch.matrix_power(cycle_matrix, self.cycle_num)
        # skip = self.skip_node
        B, M, N = cycle_matrix.shape
        target = torch.eye(M).repeat(B, 1, 1).to(cycle_matrix.device)
        _, labels = target.max(dim=-1)

        shift_offset = shift_offset.repeat_interleave(M).view(B, M)
        labels = labels - shift_offset
        labels[labels < 0] = 0
        labels[labels > (M-1)] = M - 1

        skip_node = torch.abs(shift_offset).max().int()
        if skip_node != 0 and self.skip_node:
            cycle_matrix = cycle_matrix[:, skip_node:-skip_node, :]
            labels = labels[:, skip_node:-skip_node]
        return cycle_matrix, labels

    def calculate_syn_matrix(self, aff, shift_offset):
        # import pdb; pdb.set_trace()
        B, M, N = aff.shape
        labels = torch.arange(M, device=aff.device).repeat(B, 1)
        shift_offset = shift_offset.repeat_interleave(M).view(B, M)
        labels = labels - shift_offset
        labels[labels < 0] = 0
        labels[labels > (M-1)] = M - 1

        skip_node = torch.abs(shift_offset).max().int()
        if skip_node != 0 and self.skip_node:
            aff = aff[:, skip_node:-skip_node, :].contiguous().view(-1, N)
            labels = labels[:, skip_node:-skip_node].contiguous().view(-1)
        else: 
            aff = aff.contiguous().view(-1, N)
            labels = labels.contiguous().view(-1)
        return aff, labels

    def evaluate(self, feat_dict):
        # import pdb; pdb.set_trace()
        out = self.forward(feat_dict, inference=True)
        res = {}
        res['loss'] = out['loss'].view(1, -1)
        res['crw loss'] = out['crw loss'].view(1, -1)
        res['syn loss'] = out['syn loss'].view(1, -1)

        res_crw = self.evaluate_metrics(out['pred_crw'], out['target_crw'])
        res_syn = self.evaluate_metrics(out['pred_syn'], out['target_syn'])
        res['CRW Top-1 acc'] = res_crw['Top-1 acc']
        res['CRW Top-3 acc'] = res_crw['Top-3 acc']
        res['SYN Top-1 acc'] = res_syn['Top-1 acc']
        res['SYN Top-3 acc'] = res_syn['Top-3 acc']
        if self.smooth > 0:
            res['smooth loss'] = out['smooth loss'].view(1, -1)
        return res



class VoxCelebAVITDLoss(StereoCRWAugLoss):
    # audio-visual co learning loss
    def __init__(self, args, pr, device):
        super(VoxCelebAVITDLoss, self).__init__(args, pr, device)

    def forward(self, feat_dict, inference=False):
        # import pdb; pdb.set_trace()
        feat_left = feat_dict['audio_left'] 
        feat_left = F.normalize(feat_left, p=2, dim=-1)
        feat_right = feat_dict['audio_right'] 
        feat_right = F.normalize(feat_right, p=2, dim=-1)
        feat_noaugleft = feat_dict['audio_noaug_left'] 
        feat_noaugleft = F.normalize(feat_noaugleft, p=2, dim=-1)

        aff_L2R = self.feat2mat(feat_left, feat_right)
        aff_R2L = self.feat2mat(feat_right, feat_noaugleft)

        cycle_shift_offset = torch.zeros_like(feat_dict['shift_offset'])
        cycle_matrix, crw_labels = self.calc_cycle_matrix(aff_L2R, aff_R2L, cycle_shift_offset)
        if self.bidirectional:
            cycle_matrix_R, labels_R = self.calc_cycle_matrix(aff_R2L, aff_L2R, cycle_shift_offset)
            cycle_matrix = torch.cat([cycle_matrix, cycle_matrix_R], dim=1)
            crw_labels = torch.cat([crw_labels, labels_R], dim=1)
        cycle_matrix = cycle_matrix.contiguous().view(-1, cycle_matrix.shape[-1])
        crw_labels = crw_labels.contiguous().view(-1)
        crw_loss = self.criterion(torch.log(cycle_matrix + self.eps), crw_labels.long())

        syn_aff, aff_labels = self.calculate_syn_matrix(aff_L2R, feat_dict['shift_offset'])
        syn_loss = self.criterion(syn_aff, aff_labels.long())

        smooth_loss = self.calc_smooth_loss(aff_L2R, feat_dict['delay_time'])

        loss = self.crw_rate * crw_loss + self.synthetic_rate * syn_loss + smooth_loss
        if inference:
            return {
                'pred_crw': cycle_matrix,
                'target_crw': crw_labels,
                'pred_syn': syn_aff,
                'target_syn': aff_labels,
                'loss': loss,
                'crw loss': crw_loss,
                'smooth loss': smooth_loss,
                'syn loss': syn_loss
            }
        return loss




if __name__ == '__main__':
    pass


