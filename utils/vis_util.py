import numpy as np
import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter

class Visualizer:
    """可视化工具类，用于记录和显示训练过程"""
    
    def __init__(self, opt):
        self.tf_log = opt.tf_log
        self.name = opt.name
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        
        if self.tf_log:
            self.writer = SummaryWriter(
                os.path.join(opt.checkpoints_dir, opt.name, 'logs')
            )
        
        self._init_log_file()

    def _init_log_file(self):
        with open(self.log_name, "a") as log_file:
            log_file.write(
                f'================ Training Loss ({time.strftime("%c")}) ================\n'
            )

    def _write_to_log(self, message):
        with open(self.log_name, "a") as log_file:
            log_file.write(f'{message}\n')

    def plot_current_errors(self, errors, step):
        if self.tf_log:
            for tag, value in errors.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag, values, step, bins=1000):
        if self.tf_log and values is not None:
            self.writer.add_histogram(tag, values, step, bins=bins)

    def plot_current_weights(self, net, step):
        if not self.tf_log:
            return
            
        for name, param in net.named_parameters():
            if not param.requires_grad:
                continue
                
            with torch.no_grad():
                # 记录参数分布
                param_cpu = param.cpu().numpy()
                self.log_histogram(name, param_cpu, step)
                
                # 记录梯度分布（如果存在）
                if param.grad is not None:
                    grad_cpu = param.grad.cpu().numpy()
                    self.log_histogram(f'{name}/grad', grad_cpu, step)

    def print_current_errors(self, epoch, i, errors, t):
        error_msgs = [
            f'{k}:{v:.4f}' for k, v in errors.items() if v != 0
        ]
        message = f'[Epoch{epoch}|{i}] {" | ".join(error_msgs)} ({t:.3f}s)'
        
        print(message)
        self._write_to_log(message)

    def print_line(self, message):
        print(message)
        self._write_to_log(message)
