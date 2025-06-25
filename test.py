from configs.config import Config
from dataset.dataloader import DataLoader
from model.PRSnet import create_model
import torch
import scipy.io as sio
import os
import numpy as np
import ntpath
import argparse

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Test PRS-Net')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--overrides', nargs='*', default=[],
                       help='覆盖配置参数，格式：key=value，例如：test.results_dir=./new_results')
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 处理覆盖参数
    overrides = {}
    for override in args.overrides:
        try:
            key, value = override.split('=')
            # 尝试转换为数值类型
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                # 如果转换失败，保持字符串类型
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.lower() == 'none':
                    value = None
                elif value.lower() == 'inf':
                    value = float('inf')
            overrides[key] = value
        except ValueError:
            print(f"警告：忽略无效的覆盖参数 '{override}'，格式应为 key=value")
    
    # 设置测试特定的参数
    overrides.update({
        'data.num_threads': 0,  # test code only supports num_threads = 0
        'data.batch_size': 1,   # test code only supports batch_size = 1
        'data.noshuffle': True  # no shuffle for testing
    })
    
    # 加载配置
    config = Config(args.config, mode='test', overrides=overrides)
    config.print_config()

    # 创建模型
    model = create_model(config)
    if config.data_type == 16:
        model.half()
    elif config.data_type == 8:
        model.type(torch.uint8)

    # 创建数据加载器
    data_loader = DataLoader(config)
    dataset = data_loader.load_data()

    # 创建结果目录
    save_dir = os.path.join(config.results_dir, config.name, 
                           f"{config.phase}_{config.which_epoch}")
    os.makedirs(save_dir, exist_ok=True)

    # 测试
    for i, data in enumerate(dataset):
        # 进行推理
        plane, quat = model.inference(data['voxel'])

        # 获取数据路径
        data_path = data['path'][0]
        print('[%s] process mat ... %s' % (str(i), data_path))
        
        # 加载原始mat文件
        matdata = sio.loadmat(data_path, verify_compressed_data_integrity=False)

        # 获取文件名
        short_path = ntpath.basename(data_path)
        name = os.path.splitext(short_path)[0]

        # 准备保存数据
        model_data = {
            'name': name,
            'voxel': matdata['Volume'],
            'vertices': matdata['vertices'],
            'faces': matdata['faces'],
            'sample': np.transpose(matdata['surfaceSamples']),
            'axisangle': matdata['axisangle']
        }

        # 添加预测的平面和旋转参数
        for j in range(config.num_plane):
            model_data['plane'+str(j)] = plane[j].cpu().numpy()
        for j in range(config.num_quat):
            model_data['quat'+str(j)] = quat[j].cpu().numpy()

        # 保存结果
        save_path = os.path.join(save_dir, f"{name}.mat")
        sio.savemat(save_path, model_data)
        print(f"已保存结果到 {save_path}")

if __name__ == '__main__':
    main()
