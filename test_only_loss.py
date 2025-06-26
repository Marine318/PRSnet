from configs.config import Config
from dataset.dataloader import DataLoader
from model.PRSnet import create_model
from utils.obj_utils import save_symmetry_plane, save_rotation_axis
from utils.loss import SymLoss
import torch
import scipy.io as sio
import os
import numpy as np
import ntpath
import argparse
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Test PRS-Net')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml')
    parser.add_argument('--overrides', nargs='*', default=[])
    return parser.parse_args()

def main():
    args = parse_args()
    
    overrides = {}
    for override in args.overrides:
        try:
            key, value = override.split('=')
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
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
    
    overrides.update({
        'data.num_threads': 0,
        'data.batch_size': 1,
        'data.noshuffle': True
    })
    
    config = Config(args.config, mode='test', overrides=overrides)
    config.print_config()

    model = create_model(config)
    if config.data_type == 16:
        model.half()
    elif config.data_type == 8:
        model.type(torch.uint8)

    data_loader = DataLoader(config)
    dataset = data_loader.load_data()

    base_save_dir = os.path.join(config.results_dir, config.name, 
                           f"{config.phase}_{config.which_epoch}")
    os.makedirs(base_save_dir, exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    materials_dir = os.path.join(script_dir, 'utils', 'materials')
    green_mtl = os.path.join(materials_dir, 'green.mtl')
    red_mtl = os.path.join(materials_dir, 'red.mtl')

    sym_loss = SymLoss(gridBound=config.grid_bound, gridSize=config.grid_size)

    for i, data in enumerate(dataset):
        plane, quat = model.inference(data['voxel'])
        
        print(f"\n推理结果:")
        print(f"plane 类型: {type(plane)}")
        if isinstance(plane, (list, tuple)):
            print(f"plane 列表长度: {len(plane)}")
            for i, p in enumerate(plane):
                print(f"plane[{i}] 形状: {p.shape}")
                print(f"plane[{i}] 值: {p.cpu().numpy()}")
        
        print(f"\nquat 类型: {type(quat)}")
        if isinstance(quat, (list, tuple)):
            print(f"quat 列表长度: {len(quat)}")
            for i, q in enumerate(quat):
                print(f"quat[{i}] 形状: {q.shape}")
                print(f"quat[{i}] 值: {q.cpu().numpy()}\n")

        data_path = data['path'][0]
        print('[%s] process mat ... %s' % (str(i), data_path))
        
        matdata = sio.loadmat(data_path, verify_compressed_data_integrity=False)

        short_path = ntpath.basename(data_path)
        name = os.path.splitext(short_path)[0]

        model_save_dir = os.path.join(base_save_dir, name)
        os.makedirs(model_save_dir, exist_ok=True)

        shutil.copy2(green_mtl, os.path.join(model_save_dir, 'green.mtl'))
        shutil.copy2(red_mtl, os.path.join(model_save_dir, 'red.mtl'))

        model_data = {
            'name': name,
            'voxel': matdata['Volume'],
            'vertices': matdata['vertices'],
            'faces': matdata['faces'],
            'sample': np.transpose(matdata['surfaceSamples']),
            'axisangle': matdata['axisangle']
        }

        loss_file_path = os.path.join(model_save_dir, f"{name}_sym_loss.txt")
        with open(loss_file_path, 'w') as f:
            f.write(f"Model: {name}\n")
            f.write("="*50 + "\n")
            
            points = data['sample'].cuda()
            voxel = data['voxel'].cuda()
            cp = data['cp'].cuda()
            
            print("\n最终张量维度:")
            print(f"Points shape: {points.shape}")
            print(f"Voxel shape: {voxel.shape}")
            print(f"Closest points shape: {cp.shape}")
            
            f.write("\nReflection Symmetry Losses:\n")
            f.write("-"*30 + "\n")
            for j in range(config.num_plane):
                plane_j = plane[j].view(1, 4).contiguous()
                ref_loss, _ = sym_loss(points, cp, voxel, plane=[plane_j], quat=None)
                f.write(f"Plane {j} Loss: {ref_loss.item():.6f}\n")
                model_data[f'plane{j}'] = plane[j].cpu().numpy()
            
            f.write("\nRotation Symmetry Losses:\n")
            f.write("-"*30 + "\n")
            for j in range(config.num_quat):
                quat_j = quat[j].view(1, 4).contiguous()
                _, rot_loss = sym_loss(points, cp, voxel, plane=None, quat=[quat_j])
                f.write(f"Quaternion {j} Loss: {rot_loss.item():.6f}\n")
                model_data[f'quat{j}'] = quat[j].cpu().numpy()
            
            print(f"已保存对称性损失到 {loss_file_path}")

        for j in range(config.num_plane):
            plane_j = plane[j].cpu().numpy()
            print(f"平面 {j} 参数形状: {plane_j.shape}")
            print(f"平面 {j} 参数值: {plane_j}")
            plane_obj_path = os.path.join(model_save_dir, f"plane{j}.obj")
            try:
                save_symmetry_plane(plane_obj_path, plane_j)
                print(f"已保存对称平面到 {plane_obj_path}")
            except Exception as e:
                print(f"保存平面时出错: {e}")
                print(f"平面参数: {plane_j}")
                continue

        for j in range(config.num_quat):
            quat_j = quat[j].cpu().numpy()
            axis_obj_path = os.path.join(model_save_dir, f"axis{j}.obj")
            try:
                save_rotation_axis(axis_obj_path, quat_j)
                print(f"已保存旋转轴到 {axis_obj_path}")
            except Exception as e:
                print(f"保存旋转轴时出错: {e}")
                print(f"旋转轴参数: {quat_j}")
                continue

        save_path = os.path.join(model_save_dir, f"{name}.mat")
        sio.savemat(save_path, model_data)
        print(f"已保存结果到 {save_path}")

if __name__ == '__main__':
    main()
