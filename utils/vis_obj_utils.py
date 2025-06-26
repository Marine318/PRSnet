import numpy as np
import os

def read_obj(filename):
    """读取OBJ文件，完全等效于MATLAB的readOBJ.m
    Args:
        filename: OBJ文件路径
    Returns:
        vertices: [N, 3] 顶点坐标
        faces: [M, 3] 面片索引
        uvs: [N, 2] 纹理坐标(可选)
    """
    vertices = []
    uvs = []
    faces = []
    
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
                
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            values = line.split()
            if not values:
                continue
            
            if values[0] == 'v':
                # 顶点坐标 - 确保是3D
                v = [float(x) for x in values[1:4]]
                vertices.append(v)
            elif values[0] == 'vt':
                # 纹理坐标 - 确保是2D
                vt = [float(x) for x in values[1:3]]
                uvs.append(vt)
            elif values[0] == 'f':
                # 完全按照MATLAB的方式处理面片数据
                face_line = ' '.join(values[1:])
                
                # 尝试三种格式，与MATLAB完全一致
                parts = face_line.split()
                face = []
                for part in parts[:3]:  # 只取前三个顶点
                    # 处理 v/vt/vn 格式
                    indices = part.split('/')
                    face.append(int(indices[0]))  # 只取顶点索引
                faces.append(face)
    
    # 转换为numpy数组，确保维度正确
    vertices = np.array(vertices, dtype=np.float32)  # [N, 3]
    faces = np.array(faces, dtype=np.int32)  # [M, 3]
    faces = faces - 1  # 转换为0-based索引
    uvs = np.array(uvs, dtype=np.float32) if uvs else None  # [N, 2] 或 None
    
    # 确保维度正确
    assert vertices.shape[1] == 3, "顶点必须是3D的"
    assert faces.shape[1] == 3, "面片必须是三角形"
    if uvs is not None:
        assert uvs.shape[1] == 2, "纹理坐标必须是2D的"
    
    return vertices, faces, uvs

def write_obj(*args):
    """写入OBJ文件，完全等效于MATLAB的obj_write.m
    Args:
        filename: 第一个参数是输出文件路径
        后续参数每三个一组：vertices, faces, color
        vertices: [N, 3] 顶点坐标
        faces: [M, 3] 面片索引 (0-based)
        color: 颜色名称('red', 'green', 'blue', 'none')
    """
    if len(args) < 4 or (len(args) - 1) % 3 != 0:
        raise ValueError("参数数量不正确")
        
    filename = args[0]
    mtl_filename = filename[:-4] + '.mtl'
    
    # 检查是否需要写入材质文件
    single_mesh_no_material = len(args) == 4 and args[3] == 'none'
    
    if not single_mesh_no_material:
        # 写入材质文件
        with open(mtl_filename, 'w') as f:
            for m in range((len(args) - 1) // 3):
                color = args[m*3 + 3]
                f.write(f'newmtl mtl{m+1}\n')
                f.write('Ns 0\n')
                f.write('Ka 1 1 1\n')
                
                if color == 'red':
                    f.write('Kd 0.4 0 0\n')  # 102/255 ≈ 0.4
                    f.write('d 0.8\n')
                elif color == 'green':
                    f.write('Kd 0 0.408 0.216\n')  # [0, 104/255, 55/255]
                    f.write('d 0.8\n')
                elif color == 'blue':
                    f.write('Kd 0 0.102 1\n')  # [0, 26/255, 1]
                    f.write('d 0.8\n')
                else:  # none
                    f.write('Kd 0.961 0.961 0.961\n')  # 245/255 ≈ 0.961
                
                f.write('Ks 0 0 0\n')
                f.write('Ke 0 0 0\n')
                f.write('illum 2\n\n')
    
    # 写入OBJ文件
    with open(filename, 'w') as f:
        if not single_mesh_no_material:
            f.write(f'mtllib {os.path.basename(mtl_filename)}\n\n')
        
        vertex_offset = 0
        for m in range((len(args) - 1) // 3):
            vertices = args[m*3 + 1]
            faces = args[m*3 + 2]
            color = args[m*3 + 3]
            
            # 确保维度正确
            assert vertices.shape[1] == 3, f"顶点必须是3D的，当前shape: {vertices.shape}"
            assert faces.shape[1] == 3, f"面片必须是三角形，当前shape: {faces.shape}"
            
            if not single_mesh_no_material:
                f.write(f'usemtl mtl{m+1}\n')
            
            # 写入顶点 - 完全按照MATLAB的方式
            for i in range(vertices.shape[0]):
                v = vertices[i]
                f.write('v {:.6f} {:.6f} {:.6f}\n'.format(v[0], v[1], v[2]))
            f.write('\n')
            
            # 写入面片 - 完全按照MATLAB的方式
            for i in range(faces.shape[0]):
                face = faces[i]
                # 注意：这里face已经是0-based，需要加1转换为1-based
                f.write('f {} {} {}\n'.format(
                    face[0] + 1 + vertex_offset,
                    face[1] + 1 + vertex_offset,
                    face[2] + 1 + vertex_offset
                ))
            f.write('\n')
            
            vertex_offset += vertices.shape[0] 