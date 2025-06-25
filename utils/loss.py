import torch
import torch.nn as nn
import numpy as np
from .transformers import Transformers

class RegularLoss(nn.Module):
    #正则化损失类，用于计算旋转和平面的正则化损失
    
    def __init__(self):
        super(RegularLoss, self).__init__()
        self.I = torch.eye(3).cuda()  # 3x3单位矩阵
        
    def normalize(self, x, enddim=4):
        return x/(1E-12 + torch.norm(x[:,:enddim], dim=1, p=2, keepdim=True))
        
    def _compute_orthogonal_loss(self, params, start_dim, weight):
        vectors = [self.normalize(p[:, start_dim:start_dim+3]).unsqueeze(2) for p in params]
        matrix = torch.cat(vectors, 2)
        matrix_t = torch.transpose(matrix, 1, 2)
        diff = torch.matmul(matrix, matrix_t) - self.I
        return diff.pow(2).sum(2).sum(1).mean() * weight
        
    def __call__(self, plane=None, quat=None, weight=1):
        """计算正则化损失
        Args:
            plane: 平面参数列表
            quat: 四元数参数列表
            weight: 损失权重
        Returns:
            reg_plane: 平面正则化损失
            reg_rot: 旋转正则化损失
        """
        reg_rot = torch.zeros(1, device='cuda')
        reg_plane = torch.zeros(1, device='cuda')
        
        if plane:
            reg_plane = self._compute_orthogonal_loss(plane, 0, weight)
            
        if quat:
            reg_rot = self._compute_orthogonal_loss(quat, 1, weight)
            
        return reg_plane, reg_rot

def point_closest_cell_index(points, gridBound=0.5, gridSize=32):

    cell_size = 2 * gridBound / gridSize  #这是每个网格的大小
    offset = cell_size / 2  #我让半个网格大小作为偏移
    normalized_points = (points + gridBound - offset) / cell_size
    
    return torch.round(
        torch.clamp(normalized_points, min=0, max=gridSize-1)
    )

class CalDistance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, trans_points, cp, voxel, gridSize, weight=1):
        nb = point_closest_cell_index(trans_points)
        
        grid_factors = torch.cuda.FloatTensor([gridSize**2, gridSize, 1])
        flat_indices = torch.matmul(nb, grid_factors).long()

        flat_voxel = voxel.view(-1, gridSize**3)
        mask = 1 - torch.gather(flat_voxel, 1, flat_indices)
        
        expanded_indices = flat_indices.unsqueeze(2).repeat(1,1,3)
        expanded_mask = mask.unsqueeze(2).repeat(1,1,3)
        
        #找到最近点并计算距离
        closest_points = torch.gather(cp, 1, expanded_indices)
        distance = (trans_points - closest_points) * expanded_mask
        
        ctx.save_for_backward(distance)
        ctx.constant = weight
        
        squared_dist = torch.pow(distance, 2)
        return weight * torch.mean(squared_dist.sum(dim=(1,2)))

    @staticmethod
    def backward(ctx, grad_output):
        #获取保存的距离数据
        distance = ctx.saved_tensors[0]
        batch_size = distance.shape[0]
        
        #计算梯度
        grad_trans_points = 2 * distance * ctx.constant / batch_size
        
        #只返回trans_points的梯度，其他参数不需要梯度
        return grad_trans_points, None, None, None, None

class SymLoss(nn.Module):
    #对称性损失类
    def __init__(self, gridBound, gridSize):
        super(SymLoss, self).__init__()
        self.gridSize = gridSize
        self.gridBound = gridBound
        self.cal_distance = CalDistance.apply
        self.transformers = Transformers()
        
    def _compute_transform_loss(self, points, cp, voxel, params, transform_func):
        loss = torch.zeros(1, device='cuda')
        if not params:
            return loss
            
        for param in params:
            transformed_points = transform_func(points, param)
            loss += self.cal_distance(transformed_points, cp, voxel, self.gridSize)
        return loss
        
    def __call__(self, points, cp, voxel, plane=None, quat=None, weight=1):
        ref_loss = self._compute_transform_loss(
            points, cp, voxel, plane,
            self.transformers.plane_sym_trans
        )
        
        rot_loss = self._compute_transform_loss(
            points, cp, voxel, quat,
            self.transformers.rot_sym_trans
        )
        
        return ref_loss, rot_loss
