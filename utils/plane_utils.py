import numpy as np
from scipy.spatial.transform import Rotation

def axang2rotm(axang):
    """将轴角表示转换为旋转矩阵
    
    Args:
        axang: 形状为(N,4)的数组，每行包含旋转轴[x,y,z]和旋转角度(弧度)
        
    Returns:
        R: 形状为(N,3,3)的旋转矩阵数组
        
    Example:
        axang = np.array([0, 1, 0, np.pi/2])
        R = axang2rotm(axang)
    """
    # 输入验证和预处理
    axang = np.asarray(axang, dtype=np.float64)
    if axang.ndim == 1:
        axang = axang.reshape(1, -1)
    if axang.shape[1] != 4:
        raise ValueError('Input array must have 4 columns [ax ay az theta]')
    
    # 标准化旋转轴
    v = axang[:, :3]
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = np.divide(v, norms, where=norms!=0)
    
    # 提取旋转角度
    theta = axang[:, 3]
    
    # 计算旋转矩阵的基本元素
    cth = np.cos(theta)
    sth = np.sin(theta)
    vth = (1 - cth)
    
    # 提取标准化后的轴向量分量
    vx = v[:, 0]
    vy = v[:, 1]
    vz = v[:, 2]
    
    # 构建旋转矩阵
    R = np.zeros((len(axang), 3, 3))
    
    # 第一行
    R[:, 0, 0] = vx * vx * vth + cth
    R[:, 0, 1] = vx * vy * vth - vz * sth
    R[:, 0, 2] = vx * vz * vth + vy * sth
    
    # 第二行
    R[:, 1, 0] = vy * vx * vth + vz * sth
    R[:, 1, 1] = vy * vy * vth + cth
    R[:, 1, 2] = vy * vz * vth - vx * sth
    
    # 第三行
    R[:, 2, 0] = vz * vx * vth - vy * sth
    R[:, 2, 1] = vz * vy * vth + vx * sth
    R[:, 2, 2] = vz * vz * vth + cth
    
    # 如果输入是单个轴角，返回单个3x3矩阵
    if len(axang) == 1:
        R = R[0]
        
    return R

def get_plane(plane, model, idx=None):
    """获取平面参数，完全等效于MATLAB的getplane.m
    Args:
        plane: 平面参数 [4]
        model: 包含vertices和axisangle的字典
        idx: 可选的轴向索引
    Returns:
        plane: 更新后的平面参数
        vertices: 平面顶点
        faces: 面片索引
        w1,w2,h1,h2: 平面边界
    """
    # 标准化平面参数
    plane = np.array(plane).reshape(1,-1)
    plane = plane / np.linalg.norm(plane[0,:3])
    
    # 基础面片定义
    faces = np.array([[1,2,3], [3,2,4]], dtype=np.int32) - 1  # 转换为0-based索引
    
    # 应用旋转
    R = axang2rotm(model['axisangle'])
    rot_plane = (R.T @ plane[0,:3]).reshape(1,-1)
    
    # 找到主轴
    idx_ = np.argmax(np.abs(plane[0,:3]))
    
    # 计算平面上的点
    q = np.zeros(3)
    if idx_ == 2:
        q[2] = -plane[0,3]/plane[0,2]
    elif idx_ == 1:
        q[1] = -plane[0,3]/plane[0,1]
    else:
        q[0] = -plane[0,3]/plane[0,0]
    
    rot_q = R.T @ q
    
    # 计算旋转平面参数
    d = -np.dot(rot_plane[0,:3], rot_q)
    rot_plane = np.concatenate([rot_plane, [[d]]], axis=1)
    
    # 转换顶点坐标
    points = (R.T @ model['vertices'].T).T
    
    # 根据主轴调整坐标
    if idx is None:
        idx = np.argmax(np.abs(rot_plane[0,:3]))
    
    if idx == 1:
        points = points[:,[0,2,1]]
        rot_plane = rot_plane[0,[0,2,1,3]]
    elif idx == 0:
        points = points[:,[2,1,0]]
        rot_plane = rot_plane[0,[2,1,0,3]]
    else:
        rot_plane = rot_plane[0]
    
    # 计算边界
    w1 = np.max(points[:,0]) + 0.1
    w2 = np.min(points[:,0]) - 0.1
    h1 = np.max(points[:,1]) + 0.1
    h2 = np.min(points[:,1]) - 0.1
    
    # 生成平面顶点
    a = np.array([[w1,h1], [w1,h2], [w2,h1], [w2,h2]])
    a1 = np.concatenate([a, np.ones((4,1))], axis=1)
    
    # 计算平面点 - 完全匹配MATLAB版本的计算
    # 等效于MATLAB: x=[rot_plane(1:2),rot_plane(4)]
    x = np.concatenate([rot_plane[:2], [rot_plane[3]]]).reshape(-1,1)
    v = np.concatenate([a, -a1 @ x / rot_plane[2]], axis=1)
    
    # 根据主轴还原坐标
    if idx == 1:
        v = v[:,[0,2,1]]
    elif idx == 0:
        v = v[:,[2,1,0]]
    
    # 计算最终平面参数
    v1 = v[0] - v[1]
    v2 = v[1] - v[2]
    plane = np.cross(v1, v2)
    plane = plane / np.linalg.norm(plane)
    d = -np.sum(plane * v[3])
    plane = np.append(plane, d)
    
    return plane, v, faces, w1, w2, h1, h2 