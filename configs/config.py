import os
import yaml
from typing import Dict, Any
import torch
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    
    def __init__(self, config_path: str, mode: str = 'train', overrides: Dict[str, Any] = None):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        if overrides:
            self._override_config(overrides)
            
        self.mode = mode
        self.isTrain = (mode == 'train')
        
        if isinstance(self.config['data']['max_dataset_size'], str) and \
           self.config['data']['max_dataset_size'].lower() == 'inf':
            self.config['data']['max_dataset_size'] = float('inf')
            
        self._setup_gpu()
        
        self._create_directories()
        
    def _override_config(self, overrides: Dict[str, Any]) -> None:
        for key, value in overrides.items():
            keys = key.split('.')
            config = self.config
            for k in keys[:-1]:
                config = config.setdefault(k, {})
            config[keys[-1]] = value
            
    def _setup_gpu(self) -> None:
        str_ids = self.config['base']['gpu_ids'].split(',')
        self.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.gpu_ids.append(id)
                
        if len(self.gpu_ids) > 0:
            torch.cuda.set_device(self.gpu_ids[0])
            
    def _create_directories(self) -> None:
        self.checkpoints_dir = Path(self.config['base']['checkpoints_dir'])
        self.expr_dir = self.checkpoints_dir / self.config['base']['name']
        self.expr_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.isTrain:
            self.results_dir = Path(self.config['test']['results_dir'])
            self.results_dir.mkdir(parents=True, exist_ok=True)
            
    def save(self) -> None:
        if not self.isTrain:
            return
            
        config_file = self.expr_dir / 'opt.txt'  
        with open(config_file, 'wt') as f:
            f.write('------------ Options -------------\n')
            for section, params in self.config.items():
                for k, v in sorted(params.items()):
                    f.write('%s: %s\n' % (str(k), str(v)))
            f.write('-------------- End ----------------\n')
            
    def __getattr__(self, name: str) -> Any:
        name_mapping = {
            'batchSize': 'batch_size',
            'nThreads': 'num_threads',
            'gridBound': 'grid_bound',
            'gridSize': 'grid_size',
            'input_nc': 'input_nc',
            'output_nc': 'output_nc',
            'conv_layers': 'conv_layers',
            'num_plane': 'num_plane',
            'num_quat': 'num_quat'
        }
        
        lookup_name = name_mapping.get(name, name)
        
        if lookup_name in self.config['base']:
            return self.config['base'][lookup_name]
        
        mode_config = self.config['train' if self.isTrain else 'test']
        if lookup_name in mode_config:
            return mode_config[lookup_name]
            
        for section in ['data', 'model']:
            if lookup_name in self.config[section]:
                return self.config[section][lookup_name]
                
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
    def print_config(self) -> None:
        for section, params in self.config.items():
            for k, v in sorted(params.items()):
                print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------') 