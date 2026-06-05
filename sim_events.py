# sim_events.py
import numpy as np

from config import FS

_OPTIMAL_A2 = None  # 缓存校准后的晚高峰峰值 A2


def _arrival_rate(hour, A2, weekday):
    sigma = 1.5
    lam_base = 0.05
    A1 = 0.2 * A2
    lam = (lam_base
           + A1 * np.exp(-((hour - 8.0) ** 2) / (2 * sigma ** 2))
           + A2 * np.exp(-((hour - 18.0) ** 2) / (2 * sigma ** 2)))
    if not weekday:
        lam *= 0.6
    return lam


def _next_arrival_time(t_start, A2, total_seconds, weekday, rng):
    lam_max = (A2 + 0.2 * A2 + 0.05) / 3600.0
    t = t_start
    while t < total_seconds:
        w = rng.exponential(1.0 / lam_max)
        t += w
        if t >= total_seconds:
            return None
        hour = (t % 86400) / 3600.0
        lam_t = _arrival_rate(hour, A2, weekday)
        # 注意单位统一：lam_t (辆/小时) 除以 3600 得到辆/秒
        if rng.rand() < lam_t / 3600.0 / lam_max:
            return t
    return None


def _parking_duration(rng):
    if rng.rand() < 0.7:
        mu = np.log(28800)
        sigma = 0.6
        dur = rng.lognormal(mu, sigma)
    else:
        dur = rng.exponential(5400)
    return max(dur, 60)


def _generate_raw(duration_hours, A2, fs, weekday, seed):
    rng = np.random.RandomState(seed)
    T = duration_hours * 3600
    N = int(T * fs)
    s = np.zeros(N, dtype=int)
    t = 0.0
    while t < T:
        t_arr = _next_arrival_time(t, A2, T, weekday, rng)
        if t_arr is None:
            break
        T_park = _parking_duration(rng)
        t_dep = min(t_arr + T_park, T)
        idx_arr = int(t_arr * fs)
        idx_dep = int(t_dep * fs)
        s[idx_arr:idx_dep] = 1
        t = t_dep
    return s


def calibrate_peak_rate(target_occ=0.68, duration_hours=24 * 7, seed=42):
    lo, hi = 0.1, 20.0
    best = 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        s = _generate_raw(duration_hours, mid, FS, True, seed)
        avg = s.mean()
        err = avg - target_occ
        if err > 0:
            hi = mid
        else:
            lo = mid
        best = mid
    return best


def generate_event_sequence(duration_hours, fs=FS, weekday=True,
                            target_avg_occ=0.7, seed=42):
    global _OPTIMAL_A2
    if _OPTIMAL_A2 is None:
        _OPTIMAL_A2 = calibrate_peak_rate(target_occ=target_avg_occ,
                                          duration_hours=24 * 7, seed=seed)
    return _generate_raw(duration_hours, _OPTIMAL_A2, fs, weekday, seed)


def extract_complete_durations(s, fs=FS):
    """
    从二值占用序列 s 中提取所有完整已结束的停车时长（秒）。
    忽略序列末尾仍在占用的事件。
    """
    s = np.asarray(s, dtype=int)
    diff = np.diff(s, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    durations = []
    for st, en in zip(starts, ends):
        if en < len(s) - 1:  # 只取正常结束的事件（en 对应下降沿后的 0，不是末尾截断）
            durations.append((en - st) / fs)
    return np.array(durations)


def generate_once_test():
    s = generate_event_sequence(duration_hours=24)
    print('占用率:', s.mean())  # 期望 ~0.65–0.75

    import matplotlib.pyplot as plt
    durations_hours = extract_complete_durations(s) / 3600
    plt.hist(durations_hours, bins=100, range=(0, 30), alpha=0.7, label='events duration')
    plt.axvline(5.6, color='r', linestyle='--', label='Long-stay mode (5.6 h)')
    plt.axvline(1.5, color='g', linestyle='--', label='Short-stay mean (1.5 h)')
    plt.xlabel('Duration (hours)');
    plt.ylabel('Count')
    plt.legend();
    # plt.show()
    plt.savefig('results/sanity_event_once_duration.png', bbox_inches='tight')
    plt.close()

    # 图2：完整时间序列曲线
    # x轴：采样点索引 → 小时 = index / (fs × 3600)
    plt.plot(np.arange(len(s)) / (3600 * 10), s)
    plt.xlabel('Time (hours)')
    plt.ylabel('Occupancy Status (0=Free, 1=Occupied)')
    plt.title('Parking Occupancy Time Series (7 Days)')
    plt.savefig('results/generate_event_sequence_result.png', bbox_inches='tight')
    plt.close()


def self_test():
    import random
    # 生成100个1到10000之间不重复的随机整数作为种子
    random_numbers = random.sample(range(1, 10000), 100)

    # 存储统计结果
    occupancy_rates = []  # 占用率列表
    all_durations = []  # 所有事件的停车时长（小时）

    print(f"开始生成 {len(random_numbers)} 组仿真数据...")

    for idx, seed in enumerate(random_numbers):
        # 生成7天的占用序列
        s = generate_event_sequence(duration_hours=24 * 7, fs=10, seed=seed)

        # 计算占用率
        occ_rate = s.mean()
        occupancy_rates.append(occ_rate)

        # 提取停车时长
        durations = extract_complete_durations(s) / 3600  # 转换为小时
        all_durations.extend(durations)

        # 打印进度
        if (idx + 1) % 20 == 0:
            print(f"  已完成 {idx + 1}/{len(random_numbers)} 组")

    print(f"\n统计结果:")
    print(f"  平均占用率: {np.mean(occupancy_rates):.4f} ± {np.std(occupancy_rates):.4f}")  # 期望 ~0.65–0.75
    print(f"  占用率范围: [{np.min(occupancy_rates):.4f}, {np.max(occupancy_rates):.4f}]")
    print(f"  总事件数: {len(all_durations)}")

    import matplotlib.pyplot as plt
    import pandas as pd
    # 绘制all_duration
    plt.xlabel('Duration (hours)');
    plt.ylabel('Count')
    plt.xlim(0, 40)
    data = pd.Series(all_durations)
    plt.hist(data, density=True, edgecolor='w', label='events duration', bins=40)
    data.plot(kind='kde')
    plt.legend()
    plt.savefig('results/sanity_event_duration.png', bbox_inches='tight')


if __name__ == "__main__":
    # self_test()
    generate_once_test()
