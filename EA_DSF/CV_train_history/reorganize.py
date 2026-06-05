import shutil
from pathlib import Path


def reorganize_pklot(src_root, dst_root):
    """
    将原始 PKLotSegmented 目录结构：
        src_root/UFPR04/Cloudy/2012-12-12/Occupied/xxx.jpg
    重组为：
        dst_root/UFPR04/Cloudy/Occupied/xxx.jpg
    移除中间的日期目录。
    """
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    # 遍历所有停车场目录 (UFPR04, UFPR05, PUC)
    for parking in src_root.iterdir():
        if not parking.is_dir():
            continue
        # 遍历天气目录 (Cloudy, Sunny, Rainy)
        for weather in parking.iterdir():
            if not weather.is_dir():
                continue
            # 遍历日期目录 (例如 2012-12-12)
            for date_dir in weather.iterdir():
                print("正在执行:", parking.name, weather.name, date_dir.name)
                if not date_dir.is_dir():
                    continue
                # 遍历占用状态目录 (Occupied, Empty)
                for status in date_dir.iterdir():
                    if not status.is_dir():
                        continue
                    # 目标目录：dst_root/parking/weather/status/
                    target_dir = dst_root / parking.name / weather.name / status.name
                    target_dir.mkdir(parents=True, exist_ok=True)

                    # 复制所有图片文件
                    for img_file in status.glob("*.jpg"):
                        shutil.move(img_file, target_dir / img_file.name)

    print("数据重组完成。")


if __name__ == "__main__":
    src = r"E:\DATASET\PKLot\PKLotSegmented"
    dst = r"E:\DATASET\PKLot\PKLot_organized"
    reorganize_pklot(src, dst)
