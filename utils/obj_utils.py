import numpy as np

def write_obj(filepath, vertices, faces, color='none'):
    """写入OBJ文件
    
    Args:
        filepath: 输出文件路径
        vertices: [N, 3] 顶点坐标
        faces: [M, 3] 面片索引（从1开始）
        color: 颜色名称
    """
    with open(filepath, 'w') as f:
        if color != 'none':
            f.write(f'mtllib {color}.mtl\n')
            f.write(f'usemtl {color}\n')
        
        # 写入顶点
        for v in vertices:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')
        
        # 写入面片（确保索引从1开始）
        for face in faces:
            # 将0-based索引转换为1-based
            f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')

def create_plane_mesh(plane_params, size=1.0):
    """创建平面的网格表示
    
    Args:
        plane_params: 平面参数，可以是以下格式之一：
            - [1, 4] 或 [4,] 数组: [a, b, c, d] (ax + by + cz + d = 0)
            - [1, 3] 或 [3,] 数组: [nx, ny, nz] (法向量)
            - [1, 1] 或 [1,] 数组: [d] (假设法向量为 [0, 0, 1])
        size: 平面大小
    
    Returns:
        vertices: [N, 3] 顶点坐标
        faces: [M, 3] 面片索引
    """
    # 处理输入参数
    plane_params = np.asarray(plane_params).squeeze()  # 移除多余的维度
    
    if len(plane_params.shape) == 0:
        # 如果是标量，转换为数组
        plane_params = np.array([0, 0, 1, float(plane_params)])
    elif plane_params.shape[0] == 1:
        # 如果是单个值，假设是d，法向量为[0,0,1]
        plane_params = np.array([0, 0, 1, float(plane_params[0])])
    elif plane_params.shape[0] == 3:
        # 如果是3D向量，假设是法向量，d=0
        normal = plane_params
        plane_params = np.concatenate([normal, [0.0]])
    elif plane_params.shape[0] != 4:
        raise ValueError(f"不支持的平面参数形状: {plane_params.shape}")

    # 提取平面法向量和d
    normal = plane_params[:3]
    d = plane_params[3]
    
    # 归一化法向量
    norm = np.linalg.norm(normal)
    if norm < 1e-6:  # 避免除以零
        normal = np.array([0.0, 0.0, 1.0])
    else:
        normal = normal / norm
    
    # 找到两个与法向量垂直的向量
    if abs(normal[0]) < abs(normal[1]):
        tangent1 = np.array([1.0, 0.0, 0.0])
    else:
        tangent1 = np.array([0.0, 1.0, 0.0])
    tangent1 = tangent1 - normal * np.dot(normal, tangent1)
    tangent1 = tangent1 / np.linalg.norm(tangent1)
    
    # 叉乘得到第二个切向量
    tangent2 = np.cross(normal, tangent1)
    
    # 计算平面上的一个点
    point = -d * normal
    
    # 创建平面的四个顶点
    vertices = []
    vertices.append(point + size * (tangent1 + tangent2))
    vertices.append(point + size * (tangent1 - tangent2))
    vertices.append(point + size * (-tangent1 - tangent2))
    vertices.append(point + size * (-tangent1 + tangent2))
    vertices = np.array(vertices)
    
    # 创建两个三角形面片
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    
    return vertices, faces

def create_axis_mesh(axis, angle, length=1.0, radius=0.02):
    """创建旋转轴的圆柱体网格表示
    
    Args:
        axis: [1, 4] 或 [4,] 四元数数组
        angle: 不使用，因为角度信息已包含在四元数中
        length: 轴的长度
        radius: 圆柱体半径
    
    Returns:
        vertices: [N, 3] 顶点坐标
        faces: [M, 3] 面片索引
    """
    # 从四元数中提取旋转轴
    quat = np.asarray(axis).squeeze()  # 移除多余的维度
    if len(quat) != 4:
        raise ValueError(f"四元数参数必须是4维的，得到: {quat.shape}")
    
    # 将四元数转换为旋转轴
    # 四元数格式：[x*sin(theta/2), y*sin(theta/2), z*sin(theta/2), cos(theta/2)]
    sin_theta_2 = np.linalg.norm(quat[:3])
    if sin_theta_2 < 1e-6:
        axis = np.array([0.0, 0.0, 1.0])
    else:
        axis = quat[:3] / sin_theta_2
    
    # 创建圆柱体
    segments = 16  # 圆柱体的段数
    vertices = []
    faces = []
    
    # 找到与轴垂直的两个方向
    if abs(axis[0]) < abs(axis[1]):
        tangent1 = np.array([1.0, 0.0, 0.0])
    else:
        tangent1 = np.array([0.0, 1.0, 0.0])
    tangent1 = tangent1 - axis * np.dot(axis, tangent1)
    tangent1 = tangent1 / np.linalg.norm(tangent1)
    tangent2 = np.cross(axis, tangent1)
    
    # 创建两个圆的顶点，现在从-length/2开始到length/2
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        
        # 计算顶点位置，从中心向两边延伸
        point = x * tangent1 + y * tangent2
        vertices.append(point - (length/2) * axis)  # 负方向端点
        vertices.append(point + (length/2) * axis)  # 正方向端点
    
    vertices = np.array(vertices)
    
    # 创建侧面的三角形
    for i in range(segments):
        i2 = (i + 1) % segments
        faces.append([2*i, 2*i+1, 2*i2])
        faces.append([2*i2, 2*i+1, 2*i2+1])
    
    # 创建顶部和底部的三角形
    center_bottom = len(vertices)  # 底部中心点（负方向）
    center_top = center_bottom + 1  # 顶部中心点（正方向）
    vertices = np.vstack([vertices, 
                         -(length/2) * axis,  # 底部中心点
                         (length/2) * axis])  # 顶部中心点
    
    for i in range(segments):
        i2 = (i + 1) % segments
        faces.append([center_bottom, 2*i, 2*i2])  # 底部圆面
        faces.append([center_top, 2*i2+1, 2*i+1])  # 顶部圆面
    
    faces = np.array(faces)
    
    return vertices, faces

def save_symmetry_plane(filepath, plane_params):
    """保存对称平面为OBJ文件
    
    Args:
        filepath: 输出文件路径
        plane_params: [1, 4] 或 [4,] 平面参数
    """
    vertices, faces = create_plane_mesh(plane_params)
    write_obj(filepath, vertices, faces, 'green')

def save_rotation_axis(filepath, quat_params):
    """保存旋转轴为OBJ文件
    
    Args:
        filepath: 输出文件路径
        quat_params: [1, 4] 或 [4,] 四元数参数
    """
    vertices, faces = create_axis_mesh(quat_params, None)  # 角度信息已包含在四元数中
    write_obj(filepath, vertices, faces, 'red') 