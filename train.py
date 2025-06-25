from configs.config import Config
from dataset.dataloader import DataLoader
from model.PRSnet import create_model
from utils.vis_util import Visualizer
import numpy as np
import torch
import os
import time
import argparse

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train PRS-Net')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--overrides', nargs='*', default=[],
                       help='覆盖配置参数 (key=value)')
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 处理覆盖参数
    overrides = {}
    for override in args.overrides:
        key, value = override.split('=')
        overrides[key] = value
    
    # 加载配置
    config = Config(args.config, mode='train', overrides=overrides)
    config.print_config()
    
    # 设置随机种子
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)

    # 处理断点续训
    iter_path = os.path.join(config.expr_dir, 'iter.txt')
    best_metric_path = os.path.join(config.expr_dir, 'best_metric.txt')
    
    if config.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
            if os.path.exists(best_metric_path):
                best_metric = float(np.loadtxt(best_metric_path))
            else:
                best_metric = float('inf')
        except:
            start_epoch, epoch_iter = 1, 0
            best_metric = float('inf')
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
    else:
        start_epoch, epoch_iter = 1, 0
        best_metric = float('inf')

    # 创建模型
    model = create_model(config)
    print(model)

    # 创建数据加载器
    data_loader = DataLoader(config)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('Number of training images = %d' % dataset_size)

    # 初始化训练状态
    total_steps = (start_epoch-1) * dataset_size + epoch_iter
    display_delta = total_steps % config.display_freq
    print_delta = total_steps % config.print_freq
    save_delta = total_steps % config.save_latest_freq

    # 创建可视化器
    visualizer = Visualizer(config)

    # 开始训练循环
    for epoch in range(start_epoch, config.niter + config.niter_decay + 1):
        epoch_start_time = time.time()

        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += config.batch_size
            epoch_iter += config.batch_size

            # 前向传播
            losses = model(data['voxel'], data['sample'], data['cp'])
            
            # 处理损失
            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
            losses_dict = dict(zip(model.loss_names, losses))

            # 计算总损失
            total_loss = sum(loss for loss in losses if not isinstance(loss, int))

            # 反向传播
            model.optimizer_PRS.zero_grad()
            total_loss.backward()
            model.optimizer_PRS.step()

            # 显示训练状态
            if total_steps % config.print_freq == print_delta:
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in losses_dict.items()}
                t = (time.time() - iter_start_time) / config.batch_size
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                if config.tf_log:
                    visualizer.plot_current_errors(errors, total_steps)
                    visualizer.plot_current_weights(model, total_steps)
                visualizer.print_line('')

            # 保存最新模型
            if total_steps % config.save_latest_freq == save_delta:
                print('Saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

            # 保存最佳模型
            if config.save_best:
                current_metric = total_loss.item() if config.best_metric == 'total_loss' else losses_dict[config.best_metric].item()
                if current_metric < best_metric:
                    print('Found better model (epoch %d, total_steps %d)' % (epoch, total_steps))
                    print('Metric improved from %.6f to %.6f' % (best_metric, current_metric))
                    best_metric = current_metric
                    model.save('best')
                    np.savetxt(best_metric_path, [best_metric], fmt='%.6f')

            if epoch_iter >= dataset_size:
                break

        # 结束一个epoch
        iter_end_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, config.niter + config.niter_decay, time.time() - epoch_start_time))

        # 保存当前epoch的模型
        if epoch % config.save_epoch_freq == 0:
            print('Saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save('latest')
            model.save(epoch)
            np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

        # 如果在debug模式下，只运行一个epoch
        if config.debug:
            break

if __name__ == '__main__':
    main()
