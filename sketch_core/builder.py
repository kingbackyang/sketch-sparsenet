from typing import Callable

import torch
import torch.optim
from torch import nn
from torchpack.utils.config import configs
from torchpack.utils.typing import Dataset, Optimizer, Scheduler
import numpy as np
from sketch_core.seesawloss_check import SeesawLoss


__all__ = [
    'make_dataset', 'make_model', 'make_criterion', 'make_optimizer',
    'make_scheduler'
]


class SketchLoss(nn.Module):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
    source: 源域数据（n * len(x))
    target: 目标域数据（m * len(y))
    kernel_mul:
    kernel_num: 取不同高斯核的数量
    fix_sigma: 不同高斯核的sigma值
    Return:
    loss: MMD loss
    '''
    def __init__(self, num=0):
        super(SketchLoss, self).__init__()
        self.criter = nn.CrossEntropyLoss(label_smoothing=0.5)
        # if num == 0:
        #     self.text_features = np.load("/media/kemove/403/yangjingru/spvnasaux/point0_encode.npy").astype(np.float32)
        # elif num == 5:
        #     self.text_features = np.load("/media/kemove/403/yangjingru/spvnasaux/text5_encode.npy").astype(np.float32)
        # elif num == 10:
        #     self.text_features = np.load("/media/kemove/403/yangjingru/spvnasaux/point10_encode.npy").astype(np.float32)
        self.text_features = np.load("/media/kemove/403/yangjingru/spvnasaux/text_features.npy").astype(np.float32)
        self.text_features = torch.from_numpy(self.text_features).cuda()

    def forward(self, final_feature, image_features, gt):
        celoss = self.criter(final_feature, gt)
        abs_error = 10 * image_features @ self.text_features.t()
        # abs_error = torch.abs(image_features.unsqueeze(dim=1) - self.text_features.unsqueeze(dim=0)).sum(dim=-1)
        ap = abs_error[torch.arange(gt.shape[0]), gt]
        one_hot_inv = torch.ones(final_feature.shape[0], final_feature.shape[1])
        one_hot_inv[torch.arange(final_feature.shape[0]), gt] = 0

        abs_error[torch.where(one_hot_inv==0)] = 10000

        # ans = abs_error*one_hot_inv.cuda()
        # an, _ = torch.max(abs_error, dim=1)
        # tri_loss = torch.mean(torch.clamp(an-ap+1, 0))
        an, _ = torch.min(abs_error, dim=1)
        tri_loss = torch.mean(torch.clamp(an - ap - 1, 0))

        return celoss + tri_loss
class MMDLoss(nn.Module):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
    source: 源域数据（n * len(x))
    target: 目标域数据（m * len(y))
    kernel_mul:
    kernel_num: 取不同高斯核的数量
    fix_sigma: 不同高斯核的sigma值
    Return:
    loss: MMD loss
    '''
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type
        self.criter = nn.CrossEntropyLoss(label_smoothing=0.5)

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target, mfeat, mlabel):
        if self.kernel_type == 'linear':
            return 0.1 * self.linear_mmd2(source, target) + 0.9 * self.criter(mfeat, mlabel)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            loss = 0.1 * loss + 0.9 * self.criter(mfeat, mlabel)
            return loss


class MMDLossv5(nn.Module):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
    source: 源域数据（n * len(x))
    target: 目标域数据（m * len(y))
    kernel_mul:
    kernel_num: 取不同高斯核的数量
    fix_sigma: 不同高斯核的sigma值
    Return:
    loss: MMD loss
    '''
    def __init__(self, kernel_type='linear', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLossv5, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type
        self.criter = nn.CrossEntropyLoss()

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.permute(*torch.arange(delta.ndim - 1, -1, -1)))
        return loss

    def forward(self, source, target, mfeat, mlabel):
        # return self.criter(mfeat, mlabel)
        if self.kernel_type == 'linear':
            return 0.2 * self.linear_mmd2(source, target) + 0.8 * self.criter(mfeat, mlabel)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            loss = 0.2*loss + 0.8 * self.criter(mfeat, mlabel)
            return loss


class FocalLoss(nn.Module):
    def __init__(self, ignore_index=255, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.nll_loss = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.nll_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1-pt)**self.gamma*ce_loss).mean()
        return focal_loss


def make_dataset() -> Dataset:
    if configs.dataset.name == 'img_stroke':
        from sketch_core.datasets import ImageStrokeData
        dataset = ImageStrokeData()
    elif configs.dataset.name == 'imgdata':
        from sketch_core.datasets import SketchImageData
        dataset = SketchImageData()
    elif configs.dataset.name == 'cifarimgdata':
        from sketch_core.datasets import CifarImageStrokeData
        dataset = CifarImageStrokeData()
    elif configs.dataset.name == 'transimgstrokedata':
        from sketch_core.datasets import TransImageStrokeData
        dataset = TransImageStrokeData()
    elif configs.dataset.name == 'cifarimgembeddata':
        from sketch_core.datasets import CifarImageStrokeEmbedData
        dataset = CifarImageStrokeEmbedData()
    else:
        raise NotImplementedError(configs.dataset.name)
    return dataset


def make_model() -> nn.Module:
    if configs.model.name == 'minkunet':
        from sketch_core.models import MinkUNet
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = MinkUNet(num_classes=configs.data.num_classes, cr=cr)
    elif configs.model.name == 'spvcnn':
        from sketch_core.models import SPVCNN
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = SPVCNN(num_classes=configs.data.num_classes,
                       cr=cr,
                       pres=configs.dataset.voxel_size,
                       vres=configs.dataset.voxel_size)
    elif configs.model.name == 'SD3B':
        from sketch_core.models import SD3B
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = SD3B(num_classes=configs.data.num_classes,
                       cr=cr,
                       pres=configs.dataset.voxel_size,
                       vres=configs.dataset.voxel_size)
    elif configs.model.name == 'SD3Bmerge':
        from sketch_core.models import SD3B_ResNet18Merge
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = SD3B_ResNet18Merge(num_classes=configs.data.num_classes,
                       cr=cr,
                       pres=configs.dataset.voxel_size,
                       vres=configs.dataset.voxel_size)
    elif configs.model.name == 'SD3Bmerge34':
        from sketch_core.models import SD3B_ResNet34Merge
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = SD3B_ResNet34Merge(num_classes=configs.data.num_classes,
                       cr=cr,
                       pres=configs.dataset.voxel_size,
                       vres=configs.dataset.voxel_size)
    elif configs.model.name == 'swinbv2solo':
        from sketch_core.models import SD3B_SwinSolo
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = SD3B_SwinSolo(num_classes=configs.data.num_classes,
                       cr=cr,
                       pres=configs.dataset.voxel_size,
                       vres=configs.dataset.voxel_size)
    elif configs.model.name == 'swinbv2merge':
        from sketch_core.models import SD3B_SwinMerge
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = SD3B_SwinMerge(num_classes=configs.data.num_classes,
                       cr=cr,
                       pres=configs.dataset.voxel_size,
                       vres=configs.dataset.voxel_size)
    elif configs.model.name == 'res18solo':
        from sketch_core.models import SD3B_ResNet18Solo
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = SD3B_ResNet18Solo(num_classes=configs.data.num_classes,
                       cr=cr,
                       pres=configs.dataset.voxel_size,
                       vres=configs.dataset.voxel_size)
    elif configs.model.name == 'res34solo':
        from sketch_core.models import SD3B_ResNet34Solo
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = SD3B_ResNet34Solo(num_classes=configs.data.num_classes,
                       cr=cr,
                       pres=configs.dataset.voxel_size,
                       vres=configs.dataset.voxel_size)
    elif configs.model.name == 'res50solo':
        from sketch_core.models import SD3B_ResNet50Solo
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = SD3B_ResNet50Solo(num_classes=configs.data.num_classes,
                       cr=cr,
                       pres=configs.dataset.voxel_size,
                       vres=configs.dataset.voxel_size)
    elif configs.model.name == 'res101solo':
        from sketch_core.models import SD3B_ResNet101Solo
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = SD3B_ResNet101Solo(num_classes=configs.data.num_classes,
                       cr=cr,
                       pres=configs.dataset.voxel_size,
                       vres=configs.dataset.voxel_size)
    elif configs.model.name == 'res152solo':
        from sketch_core.models import SD3B_ResNet152Solo
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = SD3B_ResNet152Solo(num_classes=configs.data.num_classes,
                       cr=cr,
                       pres=configs.dataset.voxel_size,
                       vres=configs.dataset.voxel_size)
    elif configs.model.name == 'SD3Bmerge50':
        from sketch_core.models.resnet3d import SD3B_ResNet50Merge
        model = SD3B_ResNet50Merge(num_classes=configs.data.num_classes,
                       pres=configs.dataset.voxel_size,
                       vres=configs.dataset.voxel_size)
    elif configs.model.name == 'SD3Bnorm50':
        from sketch_core.models.resnet3d import SD3B_ResNet50Norm
        model = SD3B_ResNet50Norm(num_classes=configs.data.num_classes,
                       pres=configs.dataset.voxel_size,
                       vres=configs.dataset.voxel_size)
    elif configs.model.name == 'SD3B1norm50':
        from sketch_core.models.resnet3d import SD3B1_ResNet50Norm
        model = SD3B1_ResNet50Norm(num_classes=configs.data.num_classes,
                       pres=configs.dataset.voxel_size,
                       vres=configs.dataset.voxel_size)
    elif configs.model.name == 'SD3B2norm50':
        from sketch_core.models.resnet3d import SD3B2_ResNet50Norm
        model = SD3B2_ResNet50Norm(num_classes=configs.data.num_classes,
                       pres=configs.dataset.voxel_size,
                       vres=configs.dataset.voxel_size)
    elif configs.model.name == 'Transmerge50':
        from sketch_core.models.resnet3d import Trans_ResNet50Merge
        model = Trans_ResNet50Merge(num_classes=configs.data.num_classes,
                       pres=configs.dataset.voxel_size,
                       vres=configs.dataset.voxel_size)
    elif configs.model.name == 'Transnorm50':
        from sketch_core.models.resnet3d import Trans_ResNet50Norm
        model = Trans_ResNet50Norm(num_classes=configs.data.num_classes,
                       pres=configs.dataset.voxel_size,
                       vres=configs.dataset.voxel_size)
    elif configs.model.name == 'RNNmerge50':
        from sketch_core.models.resnet3d import RNN_ResNet50Merge
        model = RNN_ResNet50Merge(num_classes=configs.data.num_classes,
                       pres=configs.dataset.voxel_size,
                       vres=configs.dataset.voxel_size)
    elif configs.model.name == 'RNNnorm50':
        from sketch_core.models.resnet3d import RNN_ResNet50Norm
        model = RNN_ResNet50Norm(num_classes=configs.data.num_classes,
                       pres=configs.dataset.voxel_size,
                       vres=configs.dataset.voxel_size)
    elif configs.model.name == 'GNNnorm50':
        from sketch_core.models.resnet3d import GNN_ResNet50Norm
        model = GNN_ResNet50Norm(num_classes=configs.data.num_classes,
                       pres=configs.dataset.voxel_size,
                       vres=configs.dataset.voxel_size)
    elif configs.model.name == 'SD3Bmerge101':
        from sketch_core.models import SD3B_ResNet101Merge, SD3B_ResNet152Merge
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = SD3B_ResNet101Merge(num_classes=configs.data.num_classes,
                       cr=cr,
                       pres=configs.dataset.voxel_size,
                       vres=configs.dataset.voxel_size)
    elif configs.model.name == 'SD3Bmerge152':
        from sketch_core.models import SD3B_ResNet152Merge
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = SD3B_ResNet152Merge(num_classes=configs.data.num_classes,
                       cr=cr,
                       pres=configs.dataset.voxel_size,
                       vres=configs.dataset.voxel_size)
    elif configs.model.name == 'mbv3merge':
        from sketch_core.models import SD3B_Mbv3Merge
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = SD3B_Mbv3Merge(num_classes=configs.data.num_classes,
                                        cr=cr,
                                        pres=configs.dataset.voxel_size,
                                        vres=configs.dataset.voxel_size)
    elif configs.model.name == 'mbv3hmerge':
        from sketch_core.models import SD3B_Mbv3HMerge
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = SD3B_Mbv3HMerge(num_classes=configs.data.num_classes,
                                        cr=cr,
                                        pres=configs.dataset.voxel_size,
                                        vres=configs.dataset.voxel_size)
    elif configs.model.name == 'effmerge':
        from sketch_core.models import SD3B_EffMerge
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = SD3B_EffMerge(num_classes=configs.data.num_classes,
                                        cr=cr,
                                        pres=configs.dataset.voxel_size,
                                        vres=configs.dataset.voxel_size)
    elif configs.model.name == 'nasmerge':
        from sketch_core.models import SD3B_NasMerge
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = SD3B_NasMerge(num_classes=configs.data.num_classes,
                                        cr=cr,
                                        pres=configs.dataset.voxel_size,
                                        vres=configs.dataset.voxel_size)
    elif configs.model.name == 'mbv2merge':
        from sketch_core.models import SD3B_Mbv2Merge
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = SD3B_Mbv2Merge(num_classes=configs.data.num_classes,
                                        cr=cr,
                                        pres=configs.dataset.voxel_size,
                                        vres=configs.dataset.voxel_size)
    elif configs.model.name == 'inv3merge':
        from sketch_core.models import SD3B_Incv3Merge
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = SD3B_Incv3Merge(num_classes=configs.data.num_classes,
                                        cr=cr,
                                        pres=configs.dataset.voxel_size,
                                        vres=configs.dataset.voxel_size)
    elif configs.model.name == 'dense169merge':
        from sketch_core.models import SD3B_DenseMerge
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = SD3B_DenseMerge(num_classes=configs.data.num_classes,
                                        cr=cr,
                                        pres=configs.dataset.voxel_size,
                                        vres=configs.dataset.voxel_size)
    elif configs.model.name == 'dense169':
        from sketch_core.models import DenseNet
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = DenseNet(num_classes=configs.data.num_classes,
                                        cr=cr,
                                        pres=configs.dataset.voxel_size,
                                        vres=configs.dataset.voxel_size)
    elif configs.model.name == 'mbv3':
        from sketch_core.models import Mbv3
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = Mbv3(num_classes=configs.data.num_classes,
                                        cr=cr,
                                        pres=configs.dataset.voxel_size,
                                        vres=configs.dataset.voxel_size)
    else:
        raise NotImplementedError(configs.model.name)
    return model


def make_criterion() -> Callable:
    if configs.criterion.name == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(
            ignore_index=configs.criterion.ignore_index, label_smoothing=0)
    elif configs.criterion.name == 'smoothcross_entropy':
        criterion = nn.CrossEntropyLoss(
            ignore_index=configs.criterion.ignore_index, label_smoothing=0.5)
    elif configs.criterion.name == 'sketch_triplet':
        criterion = SketchLoss(num=configs.criterion.num)
    elif configs.criterion.name == 'mmd_loss':
        criterion = MMDLoss(
            ignore_index=configs.criterion.ignore_index)
    elif configs.criterion.name == 'focal_loss':
        criterion = FocalLoss(
            )
    elif configs.criterion.name == 'seesaw_loss':
        criterion = SeesawLoss(num_classes=345)
    elif configs.criterion.name == 'mmd_lossv5':
        criterion = MMDLossv5(
            ignore_index=configs.criterion.ignore_index)
    else:
        raise NotImplementedError(configs.criterion.name)
    return criterion


def make_optimizer(model: nn.Module) -> Optimizer:
    if configs.optimizer.name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=configs.optimizer.lr,
                                    momentum=configs.optimizer.momentum,
                                    weight_decay=configs.optimizer.weight_decay,
                                    nesterov=configs.optimizer.nesterov)
    elif configs.optimizer.name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=configs.optimizer.lr,
            weight_decay=configs.optimizer.weight_decay)
    elif configs.optimizer.name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=configs.optimizer.lr,
            weight_decay=configs.optimizer.weight_decay)
    else:
        raise NotImplementedError(configs.optimizer.name)
    return optimizer


def make_scheduler(optimizer: Optimizer) -> Scheduler:
    if configs.scheduler.name == 'none':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=lambda epoch: 1)
    elif configs.scheduler.name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=configs.num_epochs)
    elif configs.scheduler.name == 'cosine_warmup':
        from functools import partial

        from sketch_core.schedulers import cosine_schedule_with_warmup
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=partial(cosine_schedule_with_warmup,
                              num_epochs=configs.num_epochs,
                              batch_size=configs.batch_size,
                              dataset_size=configs.data.training_size))
    else:
        raise NotImplementedError(configs.scheduler.name)
    return scheduler
