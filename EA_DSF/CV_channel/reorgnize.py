import json
import os
import random
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

# ---------- 配置 ----------

PKLOT_SRC = r"E:\DATASET\PKLot\PKLotSegmented"
PKLOT_DST = r"E:\DATASET\PKLot\PKLot_organized"
# 生成的PKLot 索引文件路径
INDEX_DIR = r"E:\DATASET\PKLot\index"
CNR_ROOT = r"E:\DATASET\CNR-EXT-Patches"
# 生成的CNR-EXT 索引文件路径
CNR_INDEX = r"E:\DATASET\CNR-EXT-Patches\index.json"

# 光照映射（单位：lux）
WEATHER_TO_LUX = {
    'Sunny': 300,
    'Cloudy': 150,
    'Rainy': 50,
    'SUNNY': 300,
    'OVERCAST': 150,
    'RAINY': 50,
}
LUX_FLUCTUATION = 0.2  # ±20% 随机波动


# ---------- 1. 重组 PKLot ----------
def reorganize_pklot():
    src = Path(PKLOT_SRC)
    dst = Path(PKLOT_DST)
    dst.mkdir(parents=True, exist_ok=True)

    for parking in src.iterdir():
        if not parking.is_dir():
            continue
        for weather in parking.iterdir():
            if not weather.is_dir():
                continue
            for date_dir in weather.iterdir():
                print("正在执行:", parking.name, weather.name, date_dir.name)
                if not date_dir.is_dir():
                    continue
                for status in date_dir.iterdir():
                    if not status.is_dir():
                        continue
                    target = dst / parking.name / weather.name / status.name
                    target.mkdir(parents=True, exist_ok=True)
                    for img in status.glob("*.jpg"):
                        shutil.move(img, target / img.name)
    print("PKLot 重组完成。")


# ---------- 2. 生成 PKLot 索引 ----------
def generate_pklot_index():
    index = []
    root = Path(PKLOT_DST)
    for parking in ['UFPR04', 'UFPR05', 'PUC']:
        parking_dir = root / parking
        if not parking_dir.exists():
            continue
        for weather_dir in parking_dir.iterdir():
            if not weather_dir.is_dir():
                continue
            weather = weather_dir.name
            base_lux = WEATHER_TO_LUX.get(weather, 150)
            print("正在执行 天气:", weather, "基准光照:", base_lux)
            for status_dir in weather_dir.iterdir():
                if not status_dir.is_dir():
                    continue
                status = status_dir.name
                label = 1 if status == 'Occupied' else 0
                for img_path in status_dir.glob("*.jpg"):
                    # 计算带波动的光照值
                    fluctuation = random.uniform(-LUX_FLUCTUATION, LUX_FLUCTUATION)
                    lux = base_lux * (1 + fluctuation)
                    index.append({
                        'path': str(img_path),
                        'parking': parking,
                        'weather': weather,
                        'label': label,
                        'lux': lux
                    })
    return index


# ---------- 3. 划分训练/验证/测试（仅 UFPR04+UFPR05） ----------
def split_index(index):
    # 过滤出 UFPR04 和 UFPR05
    train_val_index = [item for item in index if item['parking'] in ['UFPR04', 'UFPR05']]
    puc_index = [item for item in index if item['parking'] == 'PUC']

    paths = [item['path'] for item in train_val_index]
    labels = [item['label'] for item in train_val_index]
    # 分层抽样
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.2, random_state=42, stratify=labels)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)

    # 构建划分后的索引字典
    def build_subset(path_set):
        subset = []
        path_to_item = {item['path']: item for item in train_val_index}
        for p in path_set:
            subset.append(path_to_item[p])
        return subset

    train_set = build_subset(train_paths)
    val_set = build_subset(val_paths)
    test_set = build_subset(test_paths)

    return train_set, val_set, test_set, puc_index


# ---------- 4. 生成 CNR-EXT 索引 ----------
def generate_cnr_index():
    label_file = Path(CNR_ROOT) / "LABELS" / "all.txt"
    patches_dir = Path(CNR_ROOT) / "PATCHES"
    index = []
    if not label_file.exists():
        raise FileNotFoundError(f"未找到标签文件: {label_file}")

    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            rel_path, label_str = parts[0], parts[1]
            img_path = patches_dir / rel_path
            if not img_path.exists():
                continue
            # 从路径提取天气（第一级目录）
            weather = rel_path.split('/')[0]  # SUNNY / OVERCAST / RAINY
            base_lux = WEATHER_TO_LUX.get(weather, 150)
            # CNR-EXT 作为测试集，不需要额外波动，直接使用基准值
            index.append({
                'path': str(img_path),
                'weather': weather,
                'label': int(label_str),
                'lux': base_lux
            })
    return index


# ---------- 5. 保存索引 ----------
def save_index(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


# ---------- 主流程 ----------
def main():
    # 创建索引保存目录
    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)

    # 重组 PKLot（只需执行一次）
    print("正在重组 PKLot...")
    reorganize_pklot()

    # 生成完整索引
    print("正在生成 PKLot 索引...")
    full_index = generate_pklot_index()
    train_set, val_set, test_set, puc_set = split_index(full_index)

    # 保存划分后的索引
    print("正在保存索引...")
    save_index(train_set, os.path.join(INDEX_DIR, 'train.json'))
    save_index(val_set, os.path.join(INDEX_DIR, 'val.json'))
    save_index(test_set, os.path.join(INDEX_DIR, 'test.json'))
    save_index(puc_set, os.path.join(INDEX_DIR, 'puc_test.json'))

    # 生成 CNR-EXT 索引
    print("正在生成 CNR-EXT 索引...")
    cnr_index = generate_cnr_index()
    save_index(cnr_index, CNR_INDEX)

    print(f"训练集样本数: {len(train_set)}")
    print(f"验证集样本数: {len(val_set)}")
    print(f"测试集样本数: {len(test_set)}")
    print(f"PUC 测试集样本数: {len(puc_set)}")
    print(f"CNR-EXT 测试集样本数: {len(cnr_index)}")


if __name__ == "__main__":
    main()
