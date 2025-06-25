import torch
import torch.nn as nn
import numpy as np

class QuaternionUtils:
    """四元数相关的工具类"""
    def __init__(self):
        indices = torch.tensor([
            [ 0,  1,  2,  3],
            [ 1,  0,  3, -2],
            [ 2, -3,  0,  1],
            [ 3,  2, -1,  0]
        ], dtype=torch.long)
        
        self.abs_indices = torch.abs(indices)
        self.signs = torch.sign(indices.float())
    
    def hamilton_product(self, q1, q2):
        batch_size, n, _ = q1.shape
        products = []
        for i in range(4):
            q2_permuted = q2[:, :, self.abs_indices[i]] * self.signs[i].to(q2.device)
            prod = (q1 * q2_permuted).sum(dim=2, keepdim=True)
            products.append(prod)
            
        return torch.cat(products, dim=2)

    def quat_conjugate(self, quat):
        conj = quat.clone()
        conj[:, :, 1:] *= -1
        return conj

    def quat_rot_module(self, points, quats):
        points_quat = torch.cat([
            torch.zeros_like(points[:, :, :1]), 
            points
        ], dim=2)
        
        quat_conj = self.quat_conjugate(quats)
        rotated = self.hamilton_product(
            self.hamilton_product(quats, points_quat),
            quat_conj
        )
        
        return rotated[:, :, 1:4]

class Transformers:
    """几何变换工具类"""
    
    def __init__(self):
        self.quat_utils = QuaternionUtils()

    def plane_sym_trans(self, sample, plane):
        """平面对称变换
        Args:
            sample: 输入点云 [batch_size, n, 3]
            plane: 平面参数 [batch_size, 4] (ax + by + cz + d = 0)
        Returns:
            对称变换后的点云 [batch_size, n, 3]
        """
        batch_size, n_points, _ = sample.shape
        normal = plane[:, :3].unsqueeze(1).expand(-1, n_points, -1)
        d = plane[:, 3].unsqueeze(1).unsqueeze(2).expand(-1, n_points, -1)
    
        normal_norm = torch.norm(plane[:, :3], p=2, dim=1, keepdim=True).unsqueeze(1).expand(-1, n_points, -1)
        projection = (torch.sum(sample * normal, dim=2, keepdim=True) + d) / (normal_norm + 1e-5)
        reflection = 2 * projection * (normal / normal_norm)
        return sample - reflection

    def rot_sym_trans(self, sample, quat):
        """旋转对称变换
        Args:
            sample: 输入点云 [batch_size, n, 3]
            quat: 旋转四元数 [batch_size, 4]
        Returns:
            旋转后的点云 [batch_size, n, 3]
        """
        return self.rotate_module(sample, quat)

    def rotate_module(self, points, quat):
        """四元数旋转模块
        Args:
            points: 输入点云 [batch_size, n, 3]
            quat: 旋转四元数 [batch_size, 4]
        Returns:
            旋转后的点云 [batch_size, n, 3]
        """
        batch_size, n_points, _ = points.shape
        quat_expanded = quat.unsqueeze(1).expand(-1, n_points, -1)
        points_quat = torch.cat([
            torch.zeros(batch_size, n_points, 1, device=points.device),
            points
        ], dim=2)
        
        return self.quat_utils.quat_rot_module(points_quat, quat_expanded)

