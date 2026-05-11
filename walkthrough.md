# Walkthrough: PointWorld → RoboTwin HDF5 转换器

## 新增文件

所有文件放在 `robotwin_export/` 目录下，**未修改任何原始 PointWorld 代码**。

| 文件 | 用途 |
|---|---|
| [\_\_init\_\_.py](file:///c:/Users/27236/Desktop/world-action%20%20model/PointWorld/robotwin_export/__init__.py) | 包初始化 |
| [pointcloud_utils.py](file:///c:/Users/27236/Desktop/world-action%20%20model/PointWorld/robotwin_export/pointcloud_utils.py) | 深度图→点云反投影 + 随机降采样 |
| [robotwin_h5_writer.py](file:///c:/Users/27236/Desktop/world-action%20%20model/PointWorld/robotwin_export/robotwin_h5_writer.py) | RoboTwin HDF5 格式写入器 |
| [convert_droid_to_robotwin.py](file:///c:/Users/27236/Desktop/world-action%20%20model/PointWorld/robotwin_export/convert_droid_to_robotwin.py) | 主转换 CLI 脚本 |

## 数据流水线

```text
[DROID 原始 SVO 文件 (gs://...)]
          │
          ├── gather_data_dict()  ──→  连续帧 RGB (T, H, W, 3)
          │                             相机内参 (3, 3)
          │
          ├── gather_trajectory()  ──→  joint_positions (T, 7)
          │                             gripper_positions (T,)
          │                             gripper_pose (T, 7)   [xyz+quat]
          │
          └── depth/{uuid}_depth.h5 ──→ 连续帧深度 (T, H, W)
                (由 compute_depth.py 生成)
                       │
                       ▼
        ┌─────── convert_droid_to_robotwin.py ───────┐
        │                                            │
        │  1. 时间戳对齐 (canonical_timestamps)      │
        │  2. RGB resize → 480×640, JPEG 编码         │
        │  3. Depth resize → 480×640                  │
        │  4. 深度反投影 → 点云, 降采样到 2048 个点    │
        │  5. joint_positions (7D) → pad 到 14D       │
        │  6. Action[t] = State[t+1]                  │
        │  7. endpose = gripper_pose, pad 到 14D      │
        │  8. effort/velocity 填零                    │
        └────────────────────────────────────────────┘
                       │
                       ▼
               episode{N}.hdf5  (RoboTwin 格式)
```

## 输出 HDF5 格式

```
endpose:                      shape=(T, 14)      # [gripper_pose(7) | zeros(7)]
joint_action/vector:          shape=(T, 14)      # Action[t] = position[t+1]
joint_state/effort:           shape=(T, 14)      # 全零
joint_state/position:         shape=(T, 14)      # [joint_positions(7) | zeros(7)]
joint_state/vector:           shape=(T, 14)      # 同 position
joint_state/velocity:         shape=(T, 14)      # 全零
observation/head_camera/depth: shape=(T, 480, 640)
observation/head_camera/rgb:   shape=(T,)        # JPEG bytes
pointcloud:                   shape=(T, 2048, 6) # XYZ + RGB(0-255)
```

## 14 维填充方式 (DROID 单臂 → 双臂格式)

- **dims [0:7]** = Franka Panda 7 个关节角度
- **dims [7:14]** = 全零（模拟不存在的右臂）

## 使用方式

```bash
# 前置条件: 已运行 compute_depth.py 生成了 depth/ 目录
python robotwin_export/convert_droid_to_robotwin.py \
  --input real/droid_paths.txt \
  --pointworld_output_dir $DROID_ROOT \
  --output_dir /path/to/robotwin_output \
  --image_height 480 \
  --image_width 640 \
  --num_points 2048 \
  --time_skip_ratio 2 \
  --rank 0 --world_size 1
```

## 依赖

- 需要 ZED Python SDK (`pyzed`) 读取 SVO 文件
- 需要 `gsutil` 配置好用于访问 GCS 上的 DROID 数据
- 需要先运行 PointWorld 的 `compute_depth.py` 生成深度 H5 文件
