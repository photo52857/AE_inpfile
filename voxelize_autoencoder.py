import meshio
import trimesh
import numpy as np
import torch
from multiprocessing import cpu_count
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import pyvista as pv
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import re
import os
from scipy.spatial import cKDTree

class VoxelDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # 获取所有的.npy文件路径
        self.file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        voxel_matrix = np.load(self.file_list[idx])
        # 将数据转换为浮点数，并归一化（如果需要）
        voxel_matrix = voxel_matrix.astype(np.float32)
        # 归一化处理，将厚度值缩放到[0,1]范围
        voxel_matrix = (voxel_matrix - voxel_matrix.min()) / (voxel_matrix.max() - voxel_matrix.min())
        # 添加通道维度（因为PyTorch的Conv3d期望输入是[B, C, D, H, W]）
        voxel_matrix = np.expand_dims(voxel_matrix, axis=0)
        if self.transform:
            voxel_matrix = self.transform(voxel_matrix)
        return voxel_matrix
class Encoder3D(nn.Module):
    def __init__(self, latent_dim=32):
        super(Encoder3D, self).__init__()
        # 定义编码器的卷积层
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1)  # 输出: (16,24,60,116)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)  # 输出: (32,12,30,58)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)  # 输出: (64,6,15,29)
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)  # 输出: (128,3,8,15)

        # 全连接层将特征图展平并压缩到潜在向量
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 3 * 8 * 15, latent_dim)  # 128 * 3 * 8 * 15 = 46080

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (16,24,60,116)
        x = F.relu(self.conv2(x))  # (32,12,30,58)
        x = F.relu(self.conv3(x))  # (64,6,15,29)
        x = F.relu(self.conv4(x))  # (128,3,8,15)
        x = self.flatten(x)  # 展平为 (batch_size, 46080)
        latent = self.fc1(x)  # 压缩到 latent_dim
        return latent
class Decoder3D(nn.Module):
    def __init__(self, latent_dim=32):
        super(Decoder3D, self).__init__()
        # 全连接层将潜在向量映射回特征图
        self.fc1 = nn.Linear(latent_dim, 128 * 3 * 8 * 15)  # 32 -> 46080

        # 定义解码器的反卷积层，并设置不同的 output_padding
        self.deconv1 = nn.ConvTranspose3d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=(1, 0, 0)
        )  # 预计输出: (64,6,15,29)
        self.deconv2 = nn.ConvTranspose3d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1, 1, 1)
        )  # 预计输出: (32,12,30,58)
        self.deconv3 = nn.ConvTranspose3d(
            32, 16, kernel_size=3, stride=2, padding=1, output_padding=(1, 1, 1)
        )  # 预计输出: (16,24,60,116)
        self.deconv4 = nn.ConvTranspose3d(
            16, 1, kernel_size=3, stride=2, padding=1, output_padding=(1, 1, 1)
        )  # 预计输出: (1,48,120,232)

    def forward(self, x):
        x = self.fc1(x)  # (batch_size, 46080)
        x = x.view(-1, 128, 3, 8, 15)  # 重塑为 (batch_size, 128, 3, 8, 15)
        x = F.relu(self.deconv1(x))  # (64,6,15,29)
        x = F.relu(self.deconv2(x))  # (32,12,30,58) 实际: (32,12,30,59)
        x = F.relu(self.deconv3(x))  # (16,24,60,116) 实际: (16,24,60,117)
        x = torch.sigmoid(self.deconv4(x))  # (1,48,120,232) 实际: (1,48,120,234)

        # 裁剪到目标尺寸 (1,48,120,232)
        x = x[:, :, :, :120, :232]  # 深度不变，裁剪高度和宽度
        return x
class Autoencoder3D(nn.Module):
    def __init__(self):
        super(Autoencoder3D, self).__init__()
        self.encoder = Encoder3D(64)
        self.decoder = Decoder3D(64)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
def parse_shell_thickness(filename):
    thickness_dict = {}
    current_elset = None
    with open(filename, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # 解析 *SHELL SECTION
        if line.upper().startswith('*SHELL SECTION'):
            elset_match = re.search(r'ELSET=([\w]+)', line, re.IGNORECASE)
            if elset_match:
                current_elset = elset_match.group(1)
            else:
                current_elset = None
            i += 1
            # 获取厚度值
            while i < len(lines) and not lines[i].strip().startswith('*'):
                thickness_line = lines[i].strip()
                if thickness_line and not thickness_line.startswith('**'):
                    thickness_parts = thickness_line.split(',')
                    thickness = float(thickness_parts[0])
                    # 将厚度赋予对应的 ELSET
                    if current_elset:
                        thickness_dict[current_elset] = thickness
                i += 1
            continue
        i += 1
    return thickness_dict
def parse_element_sets(filename):
    elset_elements = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
    i = 0
    current_elset = None
    while i < len(lines):
        line = lines[i].strip()
        if line.upper().startswith('*ELSET'):
            elset_match = re.search(r'ELSET=([\w]+)', line, re.IGNORECASE)
            if elset_match:
                current_elset = elset_match.group(1)
                elset_elements[current_elset] = []
            else:
                current_elset = None
            # 检查是否有生成操作
            generate_match = re.search(r'GENERATE', line, re.IGNORECASE)
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('*'):
                elset_line = lines[i].strip()
                if elset_line and not elset_line.startswith('**'):
                    parts = [int(x) for x in elset_line.split(',') if x]
                    if generate_match and len(parts) == 3:
                        start, end, increment = parts
                        elset_elements[current_elset].extend(range(start, end+1, increment))
                    else:
                        elset_elements[current_elset].extend(parts)
                i += 1
            continue
        i += 1
    return elset_elements
def extract_until_end_part(input_filepath, output_filepath='processed.inp'):
    """
    从输入文件中提取从第一行到包含 "*End Part" 的所有内容，并保存到输出文件。

    参数：
    - input_filepath: str, 输入文件的路径。
    - output_filepath: str, 输出文件的路径，默认为 'processed.inp'。
    """
    try:
        with open(input_filepath, 'r') as infile, open(output_filepath, 'w') as outfile:
            for line in infile:
                outfile.write(line)
                if line.strip().lower() == '*end part':
                    print(f'已找到 "*End Part"，并将内容写入 {output_filepath}')
                    break
            else:
                print('未找到 "*End Part"。文件已全部写入。')
    except FileNotFoundError:
        print(f'错误: 文件 "{input_filepath}" 未找到。请检查文件路径。')
    except Exception as e:
        print(f'发生错误: {e}')
    return output_filepath
def read_inp(file_name):
    file_name = extract_until_end_part(file_name)
    # 替换为您的网格文件路径和名称
    mesh = meshio.read(file_name)

    # 提取点和单元信息
    points = mesh.points  # 节点坐标数组 (N, 3)

    elset_elements = parse_element_sets(file_name)

    thickness_dict = parse_shell_thickness(file_name)

    return mesh, points, elset_elements, thickness_dict
def read_inp_files(directory):
    inp_contents = []
    directory = os.path.normpath(directory)  # 规范化目录路径
    for root, dirs, files in os.walk(directory):
        root = os.path.normpath(root)  # 规范化根路径
        for file in files:
            if file.lower().endswith('.inp'):
                file_path = os.path.join(root, file)
                file_path = os.path.normpath(file_path)  # 规范化文件路径
                try:
                    inp_contents.append(file_path)
                except Exception as e:
                    print(f"无法读取文件 {file_path}: {e}")
    return inp_contents
def assign_thickness(mesh, elset_elements, thickness_dict):
    element_thickness = {}
    for elset_name, elem_ids in elset_elements.items():
        thickness = thickness_dict.get(elset_name, None)
        if thickness is not None:
            for elem_id in elem_ids:
                element_thickness[elem_id] = thickness

    # 收集所有单元及其厚度
    elements_list = []
    element_id = 1  # 假设单元 ID 从 1 开始
    for cell_block in mesh.cells:
        if cell_block.type in ['triangle', 'quad']:
            cell_type = cell_block.type
            cell_data = cell_block.data  # (n_cells, n_nodes)
            for nodes in cell_data:
                thickness = element_thickness.get(element_id, 0)
                elements_list.append({
                    'id': element_id,
                    'nodes': nodes,
                    'type': cell_type,
                    'thickness': thickness
                })
                element_id += 1

    # 构建 mesh_faces 和 mesh_thicknesses
    mesh_faces = []
    mesh_thicknesses = []

    for elem in elements_list:
        elem_type = elem['type']
        nodes = elem['nodes']
        thickness = elem['thickness']

        if elem_type == 'triangle':
            mesh_faces.append(nodes)
            mesh_thicknesses.append(thickness)
        elif elem_type == 'quad':
            # 将四边形拆分为两个三角形
            mesh_faces.append([nodes[0], nodes[1], nodes[2]])
            mesh_thicknesses.append(thickness)
            mesh_faces.append([nodes[0], nodes[2], nodes[3]])
            mesh_thicknesses.append(thickness)

    mesh_faces = np.array(mesh_faces)
    mesh_thicknesses = np.array(mesh_thicknesses)
    return mesh_faces, mesh_thicknesses
def mesh2trimeh(points, mesh_faces, mesh_thicknesses, pitch):
    tri_mesh = trimesh.Trimesh(vertices=points, faces=mesh_faces)

    # 获取网格的包围盒，用于确定体素网格的范围
    bounding_box = tri_mesh.bounds
    extruded_meshes, face_thicknesses_list = extrude_faces(tri_mesh.vertices, tri_mesh.faces, mesh_thicknesses)

    # 合并所有薄体积网格
    combined_mesh = trimesh.util.concatenate(extruded_meshes)

    # 合并所有 face_thicknesses
    face_thicknesses = np.concatenate(face_thicknesses_list)

    # 体素化
    voxelized = combined_mesh.voxelized(pitch=pitch)

    # 获取体素矩阵
    voxel_matrix = voxelized.matrix.astype(np.float32)

    # 初始化厚度矩阵
    voxel_thickness = np.zeros(voxel_matrix.shape)

    # 获取体素坐标
    indices = np.argwhere(voxel_matrix > 0)
    voxel_centers = voxelized.indices_to_points(indices)

    # 构建面片中心点的 KD 树
    face_centers = combined_mesh.triangles_center
    thicknesses = face_thicknesses
    kdtree = cKDTree(face_centers)

    # 查找每个体素中心点最近的面片
    distances, idxs = kdtree.query(voxel_centers, k=1)

    # 设置距离阈值，确保体素属于单元
    distance_threshold = pitch * np.sqrt(3) / 2  # 体素对角线的一半

    for i, (dist, idx) in enumerate(zip(distances, idxs)):
        if dist < distance_threshold:
            voxel_idx = tuple(indices[i])
            voxel_thickness[voxel_idx] = thicknesses[idx]
    voxel_thickness = voxel_thickness.astype(np.float32)
    return voxel_thickness
def extrude_faces(vertices, faces, thicknesses):
    meshes = []
    face_thicknesses = []
    for i, face in enumerate(faces):
        thickness = thicknesses[i]
        face_vertices = vertices[face]
        # 计算面法向量
        normal = np.cross(face_vertices[1] - face_vertices[0], face_vertices[2] - face_vertices[0])
        normal = normal / np.linalg.norm(normal)
        # 拉伸面，创建薄体积
        offset = normal * (thickness / 2)
        vertices_top = face_vertices + offset
        vertices_bottom = face_vertices - offset
        # 创建棱柱（六面体）
        prism_vertices = np.vstack([vertices_top, vertices_bottom])
        prism_faces = [
            [0, 1, 2],  # 顶面
            [3, 4, 5],  # 底面
            [0, 1, 4], [0, 4, 3],
            [1, 2, 5], [1, 5, 4],
            [2, 0, 3], [2, 3, 5],
        ]
        prism_mesh = trimesh.Trimesh(vertices=prism_vertices, faces=prism_faces)
        meshes.append(prism_mesh)
        # 为该 prism_mesh 的所有面赋予对应的厚度
        face_thicknesses.append(np.full(len(prism_faces), thickness))
    return meshes, face_thicknesses
def plot_trimesh(voxel_grid, pitch):
    occupied_indices = np.argwhere(voxel_grid > 0)

    # 获取体素的物理坐标
    points = occupied_indices * pitch  # 计算每个体素的坐标
    # 创建一个PolyData对象
    point_cloud = pv.PolyData(points, force_float=False)
    # 通过体素点创建立方体
    cubes = point_cloud.glyph(scale=pitch, geom=pv.Cube())
    # 创建绘图器
    plotter = pv.Plotter()
    plotter.add_mesh(cubes, color='lightblue', show_edges=True)
    plotter.show()
def auto_pad_input(voxel_tensor, kernel_sizes, strides):
    _, _, D, H, W = voxel_tensor.shape
    total_stride = np.prod(strides)
    pad_D = (total_stride - D % total_stride) % total_stride
    pad_H = (total_stride - H % total_stride) % total_stride
    pad_W = (total_stride - W % total_stride) % total_stride
    padding = (
        pad_W // 2, pad_W - pad_W // 2,
        pad_H // 2, pad_H - pad_H // 2,
        pad_D // 2, pad_D - pad_D // 2
    )
    voxel_tensor_padded = F.pad(voxel_tensor, padding)
    return voxel_tensor_padded, padding
def cnn_train(train_loader, val_loader, num_epochs):

    # voxel_tensor = torch.tensor(voxel_grid).unsqueeze(0).unsqueeze(0)  # 形状：(1, 1, D, H, W)
    # voxel_tensor_padded, padding = auto_pad_input(voxel_tensor, kernel_sizes=[3, 3, 3], strides=[2, 2, 2])

    # 创建模型实例
    model = Autoencoder3D()

    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 3. 训练过程

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 或者使用 nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    history = {
        'train_loss': [],
        'val_loss': []
    }
    loss_value = np.inf
    # 开始训练
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        start_epoch = time.time()
        for data in train_loader:
            # 将数据移动到设备
            data, padding = auto_pad_input(data, kernel_sizes=[3, 3, 3], strides=[2, 2, 2])
            inputs = data.to(device)
            # 清零梯度

            optimizer.zero_grad()
            # 正向传播
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, inputs)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 累加损失
            epoch_train_loss += loss.item()
        # 计算平均损失
        avg_train_loss = epoch_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        pth_file = 'D:\code_total\simulation_3D\crash_instance_useNPYinsteadOfTxt\data\model\Autoencoder_epoch' + str(
            epoch) + '.pth'
        torch.save(model, pth_file)
        # 验证阶段
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                data, padding = auto_pad_input(data, kernel_sizes=[3, 3, 3], strides=[2, 2, 2])
                inputs = data.to(device)
                outputs = model(inputs)
                # 计算损失
                loss = criterion(outputs, inputs)

                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        epoch_time = time.time() - start_epoch
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f}s')
        if avg_val_loss < loss_value:
            torch.save(model.state_dict(),
                       'D:\code_total\simulation_3D\crash_instance_useNPYinsteadOfTxt\data\model\Autoencoder.pth')
            torch.save(model,
                       'D:\code_total\simulation_3D\crash_instance_useNPYinsteadOfTxt\data\model\Autoencoder_full.pth')
            loss_value = avg_val_loss
        # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), history['train_loss'], label='Train Loss')
    plt.plot(range(1, num_epochs + 1), history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
if __name__ == '__main__':
    directory_path = r'D:\code_total\simulation_3D\crash_instance_useNPYinsteadOfTxt\data\dplearning_data\trainData'  # 替换为目标文件夹路径
    inp_files = read_inp_files(directory_path)
    pitch = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = r'D:\code_total\simulation_3D\crash_instance_useNPYinsteadOfTxt\data\dplearning_data\trainData'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
##########################生成体素矩阵作为autoencoder训练集###################################################################################
    for i, file_name in enumerate(inp_files, 1):
        mesh, points, elset_elements, thickness_dict = read_inp(file_name)
        mesh_faces, mesh_thicknesses = assign_thickness(mesh, elset_elements, thickness_dict)
        voxel_grid = mesh2trimeh(points, mesh_faces, mesh_thicknesses, pitch)
        voxelMatrix_path = data_dir + '\\' + file_name.split('\\')[-1].split('.inp')[0] + '.npy'
        np.save(voxelMatrix_path, voxel_grid)
    #     # plot_trimesh(voxel_grid, pitch)
###########################autoencoder训练################################################################################
    # # # 创建数据集和数据加载器
    batch_size = 64
    num_workers = int(cpu_count() * 0.8)  # 使用 80% 的 CPU 核心
    dataset = VoxelDataset(data_dir)
    dataset_size = len(dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    cnn_train(train_loader, val_loader, 300)
##############################################################################################################################





