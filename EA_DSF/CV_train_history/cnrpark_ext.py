from pathlib import Path


def collect_cnrpark_ext(root_dir, label_file='all.txt'):
    """
    从 CNR-EXT-Patches 数据集收集所有图像路径和标签。
    参数：
        root_dir: 解压后的根目录，如 "E:/DATASET/CNR-EXT-Patches"
        label_file: 标签文件名，默认为 'all.txt'，位于 LABELS 子目录下
    返回：
        image_paths: 绝对路径列表
        labels: 对应的标签列表 (0: free, 1: busy)
    """
    root_dir = Path(root_dir)
    patches_dir = root_dir / "PATCHES"
    label_path = root_dir / "LABELS" / label_file

    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    image_paths = []
    labels = []

    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            rel_img_path, label_str = parts[0], parts[1]
            img_abs_path = patches_dir / rel_img_path
            if img_abs_path.exists():
                image_paths.append(str(img_abs_path))
                labels.append(int(label_str))  # 0: free, 1: busy

    print(f"Loaded {len(image_paths)} images from {label_file}")
    return image_paths, labels
