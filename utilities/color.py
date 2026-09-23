import numpy as np
import matplotlib.cm as cm


my_colors = np.array([
    [0, 0, 0],      # 黑色 对应类别0 BackGround
    [0, 255, 0],    # 绿色 对应类别1 LV 
    [255, 255, 0],  # 黄色 对应类别2 RV
    [255, 0, 0],    # 红色 对应类别3 LA
    [0, 0, 255],    # 蓝色 对应类别4 RA
    [147,112,219],  # 紫色 对应类别5 Myo
    [0, 255, 255],  # 青色 对应类别6 AO
    [255, 165, 0]   # 橙色 对应类别7 PA
])
# LV, RV, LA, RA, Myo, AO, PA

def apply_color_map(labels):
    color_labels = my_colors[labels].astype(np.uint8)
    return color_labels

def apply_heatmap(slice_2d, vmin=0, vmax=1, cmap_name='jet'):
    normed = np.clip((slice_2d - vmin) / (vmax - vmin), 0, 1)
    cmap = cm.get_cmap(cmap_name)
    heatmap = (cmap(normed)[..., :3] * 255).astype(np.uint8)
    return heatmap
