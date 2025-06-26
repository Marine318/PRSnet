import numpy as np
from scipy.io import loadmat
from utils.vis_obj_utils import read_obj, write_obj
from utils.plane_utils import get_plane, axang2rotm
from utils.vis import vis

def main():
    result = loadmat('./results/exp/test_latest/completion.mat')
    
    part_obj = './piano_part.obj'
    v_part, f_part, _ = read_obj(part_obj)
    v_part = np.concatenate([v_part, np.ones((v_part.shape[0], 1))], axis=1)
    
    axisangle = result['axisangle'].flatten()
    R = axang2rotm(axisangle)
    vertices = (R.T @ result['vertices'].T).T
    
    plane = result['plane0'].flatten()
    plane, params_v, params_f, _, _, _, _ = get_plane(plane, result)
    
    lam = v_part @ plane.reshape(-1,1)
    points = v_part - 2 * lam * plane.reshape(1,-1)
    
    gt = np.array([0, 0, 1, 0])
    lam_gt = v_part @ gt.reshape(-1,1)
    gt_points = v_part - 2 * lam_gt * gt.reshape(1,-1)
    
    points = points[:,:3]
    gt_points = gt_points[:,:3]
    
    our_d = np.sqrt(np.sum((gt_points - points)**2, axis=1))
    our_d = our_d / 0.05
    
    vis(points, f_part, our_d, 'leg.ply')
    
    write_obj('./partial_piano.obj', vertices, result['faces'], 'none')
    
    write_obj('./partial_piano_plane.obj', params_v, params_f, 'green')

if __name__ == '__main__':
    main() 