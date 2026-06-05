# sim_ultrasonic.py
"""
超声波通道仿真器。
根据占用序列 s 生成测距信号，并可注入凝露、多径、丢包等失效模式。
占用序列由外部传入。
"""
from config import *
from sim_events import generate_event_sequence

rng_ultra = np.random.RandomState(SEED + 2)


class UltrasonicSimulator:
    def __init__(self, H=H_CEILING, sigma_d=SIGMA_D_ULTRASONIC, fs=FS, seed=SEED + 2):
        self.H = H
        self.sigma_d = sigma_d
        self.fs = fs
        self.rng = np.random.RandomState(seed)

    def generate_base_distance(self, s):
        """
        根据停车位占用状态序列s生成超声波传感器的基础测距值。

        该函数模拟超声波传感器在不同车辆类型下的测距行为：
        - 空闲状态：测距值为天花板高度 H
        - 占用状态：测距值为 H 减去车辆顶部高度

        参数:
            s (numpy.ndarray): 停车位占用状态序列，一维数组，长度为 N。
                              元素值为 1 表示占用，0 表示空闲。

        返回:
            numpy.ndarray: 包含高斯噪声的测距值序列，长度为 N，单位为米。
                          空闲时段测距值约为 H，占用时段测距值约为 H - h_top。
                          其中 h_top 根据车型分布随机采样得到。
        """
        N = len(s)
        d_true = np.full(N, self.H, dtype=float)

        # 检测占用区间
        occ_idx = np.where(s == 1)[0]
        if len(occ_idx) == 0:
            return d_true + self.rng.normal(0, self.sigma_d, N)

        s_diff = np.diff(np.concatenate([[0], s.astype(int), [0]]))
        t_on = np.where(s_diff == 1)[0]
        t_off = np.where(s_diff == -1)[0]

        events = []
        for on in t_on:
            # 找到对应的离开时间
            off_candidates = t_off[t_off > on]
            off = off_candidates[0] if len(off_candidates) > 0 else N - 1
            # 车型采样
            vt = self.rng.choice(VEHICLE_TYPE_DIST, p=[v['p'] for v in VEHICLE_TYPE_DIST])
            if vt['name'] == 'abnormal':
                h_top = self.rng.uniform(vt['h_min'], vt['h_max'])
            else:
                h_top = self.rng.normal(vt['h_mean'], vt['h_std'])
                if vt['h_max'] is not None:
                    h_top = min(h_top, vt['h_max'])
            events.append((on, off, h_top))

        for on, off, h_top in events:
            d_true[on:off] = self.H - h_top

        return d_true

    def inject_white_noise(self, d_true):
        """
        添加高斯白噪声。
        """
        return d_true + self.rng.normal(0, self.sigma_d, len(d_true))

    def inject_condensation(self, d_true, severity):
        """
            凝露 / 积灰：添加固定偏置并放大噪声
        """
        d_out = d_true.copy()
        b = 0.3 * severity / 2
        bias = self.rng.uniform(-1 * b, b)
        noise = self.rng.normal(0, self.sigma_d * severity * 1.5, size=len(d_out))
        d_out += bias + noise
        return d_out

    def inject_multipath(self, d, severity):
        """多径虚假回波：以概率 p 产生近距虚假值"""
        d_out = d.copy()

        # 生成多径掩码
        # 计算激活概率（softmax风格）
        activation_prob = 1 - np.exp(-severity)
        # 计算最大序列长度
        max_sequence_length = int(10 * severity)
        # 生成连续的多径事件序列
        mask = np.zeros(len(d), dtype=bool)

        if max_sequence_length > 0 and len(d) > 0:
            # 估计可能的事件数量（基于激活概率和数据长度）
            estimated_events = int(len(d) * activation_prob / max(1, max_sequence_length))
            estimated_events = max(estimated_events, 1)

            # 一次性生成所有事件的参数
            # 生成候选起始位置
            candidate_starts = self.rng.choice(len(d), size=estimated_events * 2, replace=False)
            candidate_starts.sort()

            # 生成事件长度（确保 low < high）
            event_min_len = max(max_sequence_length, 5)
            event_max_len = min(max_sequence_length + 1, len(d))

            # 如果最小长度 >= 最大长度，则调整为有效范围
            if event_min_len >= event_max_len:
                event_min_len = max(1, event_max_len - 1)

            event_lengths = self.rng.randint(event_min_len, event_max_len,
                                             size=len(candidate_starts))

            # 生成事件间隔
            gaps = self.rng.randint(0, max(1, max_sequence_length // 2),
                                    size=len(candidate_starts))

            # 构建事件区间
            current_pos = 0
            valid_events = []
            for start, length, gap in zip(candidate_starts, event_lengths, gaps):
                if start >= current_pos and start + length <= len(d):
                    valid_events.append((start, length))
                    current_pos = start + length + gap

            # 应用所有有效事件到掩码
            for start, length in valid_events:
                mask[start:start + length] = True

        # 分离空闲和占用索引
        idle_mask = mask & (d > 1.85)
        occ_mask = mask & (d <= 1.85)

        # 向量化生成虚假值：跨越1.85阈值的均匀分布
        n_idle = np.sum(idle_mask)
        n_occ = np.sum(occ_mask)

        if n_idle > 0:
            # 空闲时段
            d_out[idle_mask] = self.rng.uniform(0.3, 1.5, size=n_idle)
        if n_occ > 0:
            # 占用时段
            d_out[occ_mask] = self.rng.uniform(2.0, 2.8, size=n_occ)

        return d_out

    def inject_packet_loss(self, d, p):
        """丢包：返回 (信号, 掩码)"""
        missing_mask = self.rng.rand(len(d)) < p  # 生成缺失信号的掩码
        d_out = d.copy()
        d_out[missing_mask] = np.nan
        return d_out, missing_mask

    @staticmethod
    def inject_sensor_offline(d):
        """传感器完全离线"""
        mask = np.zeros(len(d), dtype=bool)
        d_out = np.full_like(d, np.nan)  # 全部设为 NaN
        return d_out, mask


def self_test():
    sim = UltrasonicSimulator()
    s = generate_event_sequence(duration_hours=24, seed=11451)
    d_true = sim.generate_base_distance(s)
    d = sim.inject_white_noise(d_true)
    # 占用样本应集中在 d ≈ 0.6–1.1 m，空闲样本应在 d ≈ 2.4 m

    # 添加退化
    d, _ = sim.inject_packet_loss(d, 0.5)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    t = np.arange(0, len(d)) / (FS * 3600)  # 转换为小时

    # 降采样：每 100 个点取一个（从 864000 降到 8640）
    downsample_rate = 100
    t_ds = t[::downsample_rate]
    s_ds = s[::downsample_rate]
    d_ds = d[::downsample_rate]

    # ===== 图 1: 全局对比（降采样后） =====
    axes[0].plot(t_ds, d_ds)
    axes[0].set_ylabel('Distance (m)', fontsize=10)
    axes[0].set_title('Global Comparison (Downsampled)', fontsize=11, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 24)
    axes[0].set_xticks(np.arange(0, 25))

    # ===== 图 2: 占用状态（降采样后） =====
    axes[1].plot(t_ds, s_ds, 'k-', linewidth=1.0)
    axes[1].set_ylabel('Status', fontsize=10)
    axes[1].set_title('Occupancy (Downsampled)', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 24)
    axes[1].set_xticks(np.arange(0, 25))
    axes[1].set_yticks([0, 1])

    plt.tight_layout()
    plt.savefig('results/sanity_ultrasonic_24h_packet_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('生成完成')


if __name__ == '__main__':
    self_test()
