import os.path
import torch
import torch.utils.data as data
import scipy.io as sio

def collect_dataset(dir):
    """收集目录下所有.mat文件的路径
    Args:
        dir: 数据目录路径
    Returns:
        包含所有.mat文件路径的列表
    """
    if not os.path.isdir(dir):
        raise AssertionError(f'{dir} is not a valid directory')
        
    return [os.path.join(root, fname) 
            for root, _, fnames in sorted(os.walk(dir))
            for fname in fnames 
            if fname.endswith('mat')]

class PRSDataLoader(data.Dataset):
    def __init__(self):
        super(PRSDataLoader, self).__init__()
        self.dataloader = None
        
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_train = os.path.join(self.root, opt.phase)
        self.train_paths = sorted(collect_dataset(self.dir_train))
        self.dataset_size = len(self.train_paths)
        
        def filter_and_collate(batch):
            #过滤无效数据并将有效数据组合成批次
            valid_samples = [sample for sample in batch if sample is not None]
            return torch.utils.data.dataloader.default_collate(valid_samples)
        
        #dataloader
        self.dataloader = torch.utils.data.DataLoader(
            self,
            batch_size=opt.batchSize,
            shuffle=not opt.noshuffle,
            num_workers=int(opt.nThreads),
            collate_fn=filter_and_collate)

    def __getitem__(self, index):
        safe_index = index % self.dataset_size
        data_path = self.train_paths[safe_index]
        
        try:
            mat_data = sio.loadmat(data_path, verify_compressed_data_integrity=False)
        except Exception as e:
            print(f"加载文件失败 {data_path}: {str(e)}")
            return None
            
        try:
            sample_points = mat_data['surfaceSamples']
            volume_data = mat_data['Volume']
            
            # 如果closestPoints为空，使用vertices作为替代
            if 'closestPoints' in mat_data and mat_data['closestPoints'].size > 0:
                closest_points = mat_data['closestPoints']
            else:
                closest_points = mat_data['vertices']
            
        except KeyError as e:
            print(f"数据字段缺失 {data_path}: {str(e)}")
            return None

        try:
            voxel_tensor = torch.from_numpy(volume_data).float().unsqueeze(0)
            sample_tensor = torch.from_numpy(sample_points).float().t()
            cp_tensor = torch.from_numpy(closest_points).float().reshape(-1, 3)
        except Exception as e:
            print(f"张量转换失败 {data_path}: {str(e)}")
            return None

        return {
            'voxel': voxel_tensor,
            'sample': sample_tensor,
            'cp': cp_tensor,
            'path': data_path
        }

    def __len__(self):
        return self.dataset_size // self.opt.batchSize * self.opt.batchSize
        
    def load_data(self):
        return self.dataloader
        
    def name(self):
        return 'PRSDataLoader...'

def DataLoader(opt):
    data_loader = PRSDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader 