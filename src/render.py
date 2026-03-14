import matplotlib
matplotlib.use('Agg') # 后台渲染，不需要弹窗
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pathlib import Path

# --- ⚙️ 骨骼拓扑配置 (关键) ---
# 这是基于通用人体骨骼的连接关系 (Parent-Child)
# 如果你的渲染结果看起来像"乱成一团的面条"，请修改这里的连接顺序
# 格式: [父关节索引, 子关节索引]
# 假设你的数据是 15 关节或类似结构 (Root, Hips, Knees, Feet, Spine, Neck, Head, Shoulders, Elbows, Hands)
SKELETON_CONNECTIONS = [
    # 躯干
    [0, 1], [1, 2], [2, 3],       # Right Leg (Root -> Hip -> Knee -> Foot)
    [0, 4], [4, 5], [5, 6],       # Left Leg
    [0, 7], [7, 8], [8, 9],       # Spine -> Neck -> Head
    # 手臂
    [8, 10], [10, 11], [11, 12],  # Right Arm (Neck -> Shoulder -> Elbow -> Hand)
    [8, 13], [13, 14], [14, 15],  # Left Arm
]

# 或者是简化的 BEAT 常用 47/52 关节的精简版连接
# 如果你发现报错 "index out of bounds"，说明你的关节数少于 16
# 这里提供一个自动容错的生成函数

def get_skeleton_connectivity(num_joints):
    """
    根据关节数量猜测连接关系 (仅作兜底，最好手动指定)
    """
    if num_joints >= 15:
        # 尝试返回标准连接，去掉越界的
        valid_conns = []
        for p, c in SKELETON_CONNECTIONS:
            if p < num_joints and c < num_joints:
                valid_conns.append([p, c])
        return valid_conns
    else:
        # 如果关节很少，可能只是简单的链式结构
        return [[i, i+1] for i in range(num_joints-1)]

def render_motion_to_mp4(motion_data, save_path, fps=15, title="Motion Preview"):
    """
    核心渲染函数
    :param motion_data: numpy array, shape [Frames, Joints, 3] 或 [Frames, Joints*3]
    :param save_path: output .mp4 path
    :param fps: frame rate
    """
    # 1. 数据形状修正
    if motion_data.ndim == 2:
        # 如果是 [T, J*3] -> 也就是平铺的，先 reshape 成 [T, J, 3]
        frames, dim = motion_data.shape
        motion_data = motion_data.reshape(frames, -1, 3)
    
    frames, num_joints, _ = motion_data.shape
    connections = get_skeleton_connectivity(num_joints)

    # 2. 计算全局包围盒 (防止画面抖动)
    # 我们找出所有帧中，动作到达的最远边界，把相机固定在那里
    all_x = motion_data[:, :, 0].flatten()
    all_y = motion_data[:, :, 1].flatten()
    all_z = motion_data[:, :, 2].flatten()
    
    # 为了视觉美观，适当放大一点边界
    pad = 0.2
    x_min, x_max = all_x.min() - pad, all_x.max() + pad
    y_min, y_max = all_y.min() - pad, all_y.max() + pad
    z_min, z_max = all_z.min() - pad, all_z.max() + pad
    
    # 强制让坐标轴比例一致 (防止人变扁)
    max_range = np.array([x_max-x_min, y_max-y_min, z_max-z_min]).max() / 2.0
    mid_x = (x_max+x_min) * 0.5
    mid_y = (y_max+y_min) * 0.5
    mid_z = (z_max+z_min) * 0.5

    # 3. 初始化画布 (暗黑风格)
    fig = plt.figure(figsize=(8, 6), dpi=100) # 800x600 resolution
    # 背景设为深色
    fig.patch.set_facecolor('#0b0f19') 
    
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0b0f19')
    
    # 去除坐标轴背景和刻度
    ax.grid(False)
    ax.set_axis_off()
    # 如果你想留一点地板网格感，可以保留下面这行，否则注释掉
    # ax.xaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
    
    # 初始化线条和点
    # joints: 红色点
    scat = ax.scatter([], [], [], c='#ff3366', s=15, depthshade=True) 
    # bones: 青色线
    lines = [ax.plot([], [], [], color='#00f2ea', linewidth=2)[0] for _ in connections]

    # 设置相机视角 (Elevation, Azimuth) - 根据你的坐标系调整
    # 通常 BVH 是 Y-up，Matplotlib 默认 Z-up。
    # 这里我们手动把数据喂进去的时候做交换，或者在这里调视角
    ax.view_init(elev=10, azim=45) 

    def init():
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        return lines + [scat]

    def update(frame_idx):
        # 获取当前帧姿态
        pose = motion_data[frame_idx] # [J, 3]
        
        # --- 坐标系校准 ---
        # Matplotlib 的 3D 坐标系：Z 是垂直向上的
        # 如果你的数据 Y 是向上的，需要在这里交换一下
        # 假设数据是 (x, y, z)，我们画的时候用 (x, z, y)
        xs = pose[:, 0]
        ys = pose[:, 2] # 交换 Y 和 Z
        zs = pose[:, 1] 
        
        # 更新关节(点)位置
        scat._offsets3d = (xs, ys, zs)
        
        # 更新骨骼(线)位置
        for line, (p_idx, c_idx) in zip(lines, connections):
            line.set_data([xs[p_idx], xs[c_idx]], [ys[p_idx], ys[c_idx]])
            line.set_3d_properties([zs[p_idx], zs[c_idx]])
            
        return lines + [scat]

    # 4. 开始渲染
    print(f"[Render] Generating video ({frames} frames)...")
    anim = animation.FuncAnimation(
        fig, update, frames=frames, init_func=init, blit=False, interval=1000/fps
    )
    
    # 保存
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(save_path, writer='ffmpeg', fps=fps, codec='h264')
    plt.close()
    print(f"[Render] Saved to: {save_path}")
    return save_path

# --- 测试代码 ---
if __name__ == "__main__":
    # 生成一段假数据测试一下
    # T=60帧, J=16关节, 3维
    dummy_motion = np.random.randn(60, 16, 3) * 0.1
    # 让它稍微动一动（沿Y轴上升）
    dummy_motion[:, :, 1] += np.linspace(0, 1, 60)[:, None]
    
    render_motion_to_mp4(dummy_motion, "test_render.mp4", fps=15)