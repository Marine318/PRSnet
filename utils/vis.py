import numpy as np
from PIL import Image

def vis(vertices, faces, distance, target_name):

    rgb = np.array(Image.open('./plottingscale.png'))
    
    point_count = vertices.shape[0]
    mesh_count = faces.shape[0]
    new_obj = np.zeros((3, point_count))
    color_arr = np.zeros((3, point_count), dtype=np.uint8)
    
    for i in range(point_count):
        p = vertices[i]
        
        y = int(distance[i] * 1024)
        y = max(0, min(1023, y))
        
        new_obj[:,i] = p
        color_arr[:,i] = rgb[y,0]
    
    with open(target_name, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {point_count}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('property uchar alpha\n')
        f.write(f'element face {mesh_count}\n')
        f.write('property list uint8 int32 vertex_index\n')
        f.write('end_header\n')
        
        for i in range(point_count):
            p = new_obj[:,i]
            c = color_arr[:,i]
            f.write(f'{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]} 255\n')
        
        for i in range(mesh_count):
            f.write(f'3 {faces[i,0]} {faces[i,1]} {faces[i,2]}\n') 