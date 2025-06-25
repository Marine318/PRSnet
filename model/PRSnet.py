import os
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from utils.loss import RegularLoss, SymLoss
import sys

class Frontnet(nn.Module):
    #前端特征提取网络
    def __init__(self, input_nc, output_nc, conv_layers, activation=nn.LeakyReLU(0.2, True)):
        super(Frontnet, self).__init__()
        
        channels = [input_nc]
        curr_channels = output_nc
        for _ in range(conv_layers):
            channels.append(curr_channels)
            curr_channels *= 2
            
        layers = []
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            layers.extend([
                nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.MaxPool3d(2),
                activation
            ])
            
        self.model = nn.Sequential(*layers)
        
    def forward(self, input):
        return self.model(input)

class Backnet(nn.Module):
    #后端网络：预测对称平面和旋转参数
    def __init__(self, input_nc, num_plane, num_quat, biasTerms, activation=nn.LeakyReLU(0.2, True)):
        super(Backnet, self).__init__()
        self.num_quat = num_quat
        self.num_plane = num_plane
        
        def create_mlp(bias_key, zero_weights=False):
            hidden_dims = [input_nc, input_nc//2, input_nc//4]
            layers = []
            
            for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
                layers.extend([
                    nn.Linear(in_dim, out_dim),
                    activation
                ])
            
            final_layer = nn.Linear(hidden_dims[-1], 4)
            if zero_weights:
                final_layer.weight.data.zero_()
            final_layer.bias.data = torch.Tensor(biasTerms[bias_key])
            layers.append(final_layer)
            
            return nn.Sequential(*layers)
        
        #四元数
        for i in range(self.num_quat):
            layer_name = f'quatLayer{i+1}'
            setattr(self, layer_name, create_mlp(f'quat{i+1}'))
            
        #平面
        for i in range(self.num_plane):
            layer_name = f'planeLayer{i+1}'
            setattr(self, layer_name, create_mlp(f'plane{i+1}', zero_weights=True))

    def forward(self, feature):
        """前向传播
        Returns:(quat, plane): 四元数和平面参数的元组
        """
        feature = feature.view(feature.size(0), -1)
        quat = [self.normalize(getattr(self, f'quatLayer{i+1}')(feature))
               for i in range(self.num_quat)]
        plane = [self.normalize(getattr(self, f'planeLayer{i+1}')(feature), 3)
               for i in range(self.num_plane)]
        return quat, plane
        
    def normalize(self, x, enddim=4):
        #归一化函数
        x = x/(1E-12 + torch.norm(x[:,:enddim], dim=1, p=2, keepdim=True))
        return x

class PRSNetModel(nn.Module):
    def __init__(self):
        super(PRSNetModel, self).__init__()

    def initialize(self, opt):
        self._init_basic_settings(opt)
        biasTerms = self._init_bias_terms(opt.num_plane, opt.num_quat)
        self._create_and_load_network(opt, biasTerms)
        if self.isTrain:
            self._init_training_settings(opt)
    
    def _init_basic_settings(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        
        if not opt.isTrain:
            torch.backends.cudnn.benchmark = True
            
    def _init_bias_terms(self, num_plane, num_quat):
        biasTerms = {
            'plane1': [1,0,0,0],
            'plane2': [0,1,0,0],
            'plane3': [0,0,1,0],
            'quat1': [0, 0, 0, np.sin(np.pi/2)],
            'quat2': [0, 0, np.sin(np.pi/2), 0],
            'quat3': [0, np.sin(np.pi/2), 0, 0]
        }
        
        for i in range(4, num_plane + 1):
            plane = np.random.random_sample((3,))
            biasTerms[f'plane{i}'] = (plane/np.linalg.norm(plane)).tolist() + [0]
            
        for i in range(4, num_quat + 1):
            quat = np.random.random_sample((4,))
            biasTerms[f'quat{i}'] = (quat/np.linalg.norm(quat)).tolist()
        return biasTerms
        
    def _create_and_load_network(self, opt, biasTerms):
        #创网络
        self.netPRS = self._define_network(opt, biasTerms)
        
        #预训练模型
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self._load_network(self.netPRS, self.name(), opt.which_epoch, pretrained_path)
            
    def _init_training_settings(self, opt):
        self.sym_loss = SymLoss(opt.gridBound, opt.gridSize)
        self.reg_loss = RegularLoss()
        self.loss_names = ['ref', 'rot', 'reg_plane', 'reg_rot']
        params = list(self.netPRS.parameters())
        self.optimizer_PRS = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def _define_network(self, opt, biasTerms):
        net = None
        if len(opt.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            
        activation = None
        if opt.activation == 'relu':
            activation = nn.ReLU()
        elif opt.activation == 'tanh':
            activation = nn.Tanh()
        elif opt.activation == 'lrelu':
            activation = nn.LeakyReLU(0.2, True)
            
        encoder = Frontnet(opt.input_nc, opt.output_nc, opt.conv_layers, activation)
        pred = Backnet(opt.output_nc*(2**(opt.conv_layers-1)), opt.num_plane, opt.num_quat, biasTerms, activation)
        net = nn.Sequential(encoder, pred)
        
        if len(opt.gpu_ids) > 0:
            net.cuda(opt.gpu_ids[0])
            net = torch.nn.DataParallel(net, opt.gpu_ids)
        return net

    def _save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda()

    def _load_network(self, network, network_label, epoch_label, save_dir=''):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(save_dir or self.save_dir, save_filename)
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            return
        try:
            network.load_state_dict(torch.load(save_path))
            return
        except:
            return self._handle_partial_load(network, save_path)
    def _handle_partial_load(self, network, save_path):
        pretrained_dict = torch.load(save_path)
        model_dict = network.state_dict()
        try:
            compatible_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            network.load_state_dict(compatible_dict)
            if self.opt.verbose:
                print('Pretrained network has excessive layers; Only loading layers that are used')
            return
        except:
            not_initialized = set()
            for k, v in model_dict.items():
                if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                    not_initialized.add(k.split('.')[0])
                else:
                    model_dict[k] = pretrained_dict[k]
                    
            print('Following layers are not initialized:', sorted(not_initialized))
            network.load_state_dict(model_dict)

    def name(self):
        return 'PRSNet'
    
    def forward(self, voxel, points, cp):
        voxel = Variable(voxel.data.cuda(), requires_grad=True)
        points = Variable(points.data.cuda())
        cp = Variable(cp.data.cuda())
        
        quat, plane = self.netPRS(voxel)
        loss_ref, loss_rot = self.sym_loss(points, cp, voxel, plane=plane, quat=quat)
        loss_reg_plane, loss_reg_rot = self.reg_loss(plane=plane, quat=quat, weight=self.opt.weight)
        
        return [loss_ref, loss_rot, loss_reg_plane, loss_reg_rot]

    def inference(self, voxel):
        if len(self.gpu_ids) > 0:
            voxel = Variable(voxel.data.cuda())
        else:
            voxel = Variable(voxel.data)
            
        self.netPRS.eval()
        with torch.no_grad():
            quat, plane = self.netPRS(voxel)
        return plane, quat

    def save(self, which_epoch):
        self._save_network(self.netPRS, self.name(), which_epoch, self.gpu_ids)

class InferenceModel(PRSNetModel):
    def forward(self, voxel):
        return self.inference(voxel)

def create_model(opt):
    #创建模型的工厂函数
    if opt.isTrain:
        model = PRSNetModel()
    else:
        model = InferenceModel()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))
    return model
