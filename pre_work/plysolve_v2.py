import numpy as np
from plyfile import PlyData, PlyElement

def save_colored_pc_ply(coords, colors, path):
    '''
    保存点云数据为Ply格式文件

    :param coords: 点的坐标
    :param colors: 归一化后的颜色值 (范围 0-1)
    :param path: 保存路径
    '''
    vertices = np.empty(coords.shape[0],
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                               ('red', 'uint8'), ('green', 'uint8'), ('blue', 'uint8')
                               ])
    vertices['x'] = coords[:, 0].astype('f4')
    vertices['y'] = coords[:, 1].astype('f4')
    vertices['z'] = coords[:, 2].astype('f4')

    # 将颜色值从 0-1 转换为 0-255
    vertices['red'] = (colors[:, 0] * 255).astype('uint8')
    vertices['green'] = (colors[:, 1] * 255).astype('uint8')
    vertices['blue'] = (colors[:, 2] * 255).astype('uint8')

    ply = PlyData([PlyElement.describe(vertices, 'vertex')], text=False)
    ply.write(path)

def read_ply_xyz(file):
    '''
    读取Ply文件中的点坐标 (x, y, z)

    :param file: Ply文件路径
    :return: 点坐标数组
    '''
    ply = PlyData.read(file)
    vtx = ply['vertex']

    xyz = np.stack([vtx['x'], vtx['y'], vtx['z']], axis=-1)
    return xyz

def read_ply_xyzrgb(file):
    '''
    读取Ply文件中的点坐标 (x, y, z) 和颜色 (r, g, b)

    :param file: Ply文件路径
    :return: 点坐标数组和颜色数组
    '''
    ply = PlyData.read(file)
    vtx = ply['vertex']

    xyz = np.stack([vtx['x'], vtx['y'], vtx['z']], axis=-1)
    rgb = np.stack([vtx['red'], vtx['green'], vtx['blue']], axis=-1)

    return xyz, rgb

# 主函数，完成读取 ply 文件并保存为 npy 文件
def convert_ply_to_npy(input_ply, output_npy):
    '''
    读取ply文件，归一化颜色值，保存为npy文件

    :param input_ply: 输入的ply文件路径
    :param output_npy: 输出的npy文件路径
    '''
    # 读取点坐标和颜色
    xyz, rgb = read_ply_xyzrgb(input_ply)

    # 将颜色归一化到 0-1
    rgb_normalized = rgb / 255.0

    # 合并坐标和颜色
    data = np.concatenate([xyz, rgb_normalized], axis=-1)

    # 保存为npy文件
    np.save(output_npy, data)

# 调用主函数
import os
if __name__ == "__main__":
    input_path = "./data-ply"
    output_path = "./data-npy"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 遍历输入目录下的所有 .ply 文件
    for file_name in os.listdir(input_path):
        if file_name.endswith(".ply"):
            input_file = os.path.join(input_path, file_name)
            output_file = os.path.join(output_path, file_name.replace(".ply", ".npy"))
            if not os.path.exists(output_file):
                # 转换并保存
                print(f"Processing {input_file} -> {output_file}")
                convert_ply_to_npy(input_file, output_file)

    print("All files have been processed.")
