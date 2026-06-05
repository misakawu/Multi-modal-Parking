# sim_geomag.py
"""
地磁通道仿真器。
基于地球背景磁场、传感器噪声与漂移，叠加车辆磁偶极子扰动量。
占用序列 s(t) 由外部传入（通常来自 sim_events.generate_event_sequence）。
"""
from config import *
from sim_events import generate_event_sequence

rng_geomag = np.random.RandomState(SEED + 1)


class GeomagSimulator:
    def __init__(self, B_earth=None, sigma_n=SIGMA_N, sigma_drift=SIGMA_DRIFT,
                 fs=FS, seed=SEED):
        self.B_earth = B_earth if B_earth is not None else B_EARTH.astype(float)
        self.sigma_n = sigma_n
        self.sigma_drift = sigma_drift
        self.fs = fs
        self.rng = np.random.RandomState(seed)

    def generate_baseline(self, T_seconds):
        """生成无车的基线信号（含地球磁场、漂移、噪声）"""
        N = int(T_seconds * self.fs)
        dt = 1.0 / self.fs
        drift = np.zeros(3)
        B = np.zeros((N, 3))
        for i in range(N):
            drift += self.rng.normal(0, self.sigma_drift * np.sqrt(dt), 3)
            noise = self.rng.normal(0, self.sigma_n, 3)
            B[i] = self.B_earth + drift + noise
        return B

    def inject_self_occupancy(self, B_base, s):
        """
            根据停车位占用状态序列叠加本车位车辆产生的磁扰动信号。

            该函数基于磁偶极子模型模拟车辆对地磁场的扰动效应，包括：
            - 车辆磁矩的随机采样（符合实际车辆磁场分布特性）
            - 磁偶极子场的精确计算（使用国际单位制物理常数）
            - 车辆进入和离开时的瞬态响应建模（指数上升/衰减）
            - 扰动幅值保护机制（确保信号可检测性）

            参数:
                B_base (numpy.ndarray): 基线地磁信号，形状为 (N, 3) 的二维数组。
                                       N 为采样点数，3 表示三轴磁场分量 (Bx, By, Bz)。
                                       单位为微特斯拉 (μT)。
                s (numpy.ndarray): 停车位占用状态序列，一维数组，长度为 N。
                                  元素值为 1 表示占用，0 表示空闲。
                μ₀ = 4π × 10⁻⁷ H/m (精确值)
            返回:
                numpy.ndarray: 叠加车辆磁扰动后的地磁信号，形状为 (N, 3)，单位为 μT。
                              包含基线信号、车辆扰动及其瞬态响应。
        """
        N = B_base.shape[0]
        B_signal = B_base.copy().astype(np.float64)

        # 1. 检测事件边界
        s_diff = np.diff(np.concatenate([[0], s.astype(int), [0]]))
        t_on = np.where(s_diff == 1)[0]
        t_off = np.where(s_diff == -1)[0]

        # 2. 生成每个事件的物理参数
        events = []
        for on_idx in t_on:
            # 磁矩幅值：截断正态分布，单位 A·m²
            m0 = np.clip(self.rng.normal(M0_MEAN, M0_STD), M0_MIN, M0_MAX)

            # 磁矩方向：限制在竖直方向±60°范围内（符合车辆磁场特性）
            theta = np.arccos(self.rng.uniform(0.5, 1.0))  # 极角
            phi = self.rng.uniform(0, 2 * np.pi)  # 方位角
            e = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ], dtype=np.float64)
            m = m0 * e  # 磁矩矢量 (A·m²)

            # 几何参数：传感器到车辆磁中心的距离
            r_mag = self.rng.uniform(0.4, 0.8)  # 距离范围 0.4-0.8 m

            # 【新增】物理约束：避免小磁矩 + 远距离的不利组合
            # 当 m0 < 1.0 时，限制最大距离，确保扰动可检测
            if m0 < 1.0 and r_mag > 0.6:
                r_mag = self.rng.uniform(0.4, 0.6)

            r_vec = np.array([0.0, 0.0, -1.0], dtype=np.float64) * r_mag
            r_hat = r_vec / r_mag  # 单位方向向量（竖直向下）

            # 计算磁偶极子场（使用精确物理常数）
            # μ₀ = 4π × 10⁻⁷ H/m (真空磁导率，国际单位制定义值)
            mu0_over_4pi = 1e-7  # 精确值：μ₀/(4π) = 10⁻⁷ H/m

            # 磁偶极子公式：B = (μ₀/4πr³)[3(m·r̂)r̂ - m]
            # 结果单位：Tesla (T)
            m_dot_rhat = np.dot(m, r_hat)
            B_dip_T = (mu0_over_4pi / r_mag ** 3) * (3.0 * m_dot_rhat * r_hat - m)

            # 转换为微特斯拉 (μT)：1 T = 10⁶ μT
            B_dip_uT = B_dip_T * 1e6

            # 【新增】扰动幅值保护：确保 Z 分量不低于可检测阈值
            B_dip_z_mag = abs(B_dip_uT[2])
            min_detectable = 0.8  # 最小可检测 Z 分量 (μT)，需 > σ_n=0.3
            if B_dip_z_mag < min_detectable:
                scale = min_detectable / B_dip_z_mag
                B_dip_uT = B_dip_uT * scale

            # 瞬态时间常数（转换为采样点数）
            tau_on = self.rng.uniform(0.5, 2.0) * self.fs
            tau_off = self.rng.uniform(0.5, 2.0) * self.fs

            events.append({
                't_start': int(on_idx),
                'B_dip': B_dip_uT.astype(np.float64),
                'tau_on': float(tau_on),
                'tau_off': float(tau_off),
                't_off': None
            })

        # 3. 关联离开时间（按时间顺序匹配）
        ptr = 0
        for ev in events:
            while ptr < len(t_off) and t_off[ptr] <= ev['t_start']:
                ptr += 1
            if ptr < len(t_off):
                ev['t_off'] = int(t_off[ptr])
                ptr += 1
            else:
                ev['t_off'] = N - 1

        # 4. 计算扰动场（向量化计算，保证数值精度）
        B_perturb = np.zeros((N, 3), dtype=np.float64)

        for idx, ev in enumerate(
                events_sorted if (events_sorted := sorted(events, key=lambda x: x['t_start'])) else events):
            start = ev['t_start']
            end = ev['t_off']
            B_dip = ev['B_dip']
            tau_on = ev['tau_on']
            tau_off = ev['tau_off']

            # 确定衰减终点：下一个事件开始前或信号末尾
            if idx + 1 < len(events):
                decay_end = events[idx + 1]['t_start']
            else:
                decay_end = N

            # 占用期：指数上升至稳态
            # e(t) = 1 - exp(-(t - t_on) / τ_on), t ∈ [t_on, t_off)
            occupy_len = end - start
            if occupy_len > 0:
                dt_on = np.arange(occupy_len, dtype=np.float64)
                weight_on = 1.0 - np.exp(-dt_on / tau_on)
                B_perturb[start:end] += B_dip[np.newaxis, :] * weight_on[:, np.newaxis]

            # 衰减期：指数衰减至零
            # e(t) = exp(-(t - t_off) / τ_off), t ∈ [t_off, next_t_on)
            decay_len = decay_end - end
            if decay_len > 0:
                dt_off = np.arange(decay_len, dtype=np.float64)
                weight_off = np.exp(-dt_off / tau_off)
                B_perturb[end:decay_end] += B_dip[np.newaxis, :] * weight_off[:, np.newaxis]

        return B_signal + B_perturb

    @staticmethod
    def _compute_dipole_projection(r_vector, m_direction=None):
        """
        计算磁偶极子在给定方向的投影系数
        :param r_vector: 从源到传感器的相对位置向量 (3,)
        :param m_direction: 源磁矩方向单位向量 (3,), 默认沿x轴
        :return: 投影系数标量
        """
        if m_direction is None:
            m_direction = np.array([1.0, 0.0, 0.0])  # 假设车辆沿x轴

        r_norm = np.linalg.norm(r_vector)
        r_hat = r_vector / r_norm

        # 偶极子场方向因子: [3(m·r̂)r̂ - m]
        m_dot_r = np.dot(m_direction, r_hat)
        dipole_factor = 3 * m_dot_r * r_hat - m_direction

        # 投影到本地传感器方向（假设传感器z轴朝上）
        sensor_direction = np.array([0.0, 0.0, 1.0])
        proj = np.abs(np.dot(dipole_factor, sensor_direction))

        return np.clip(proj, 0.1, 2.0)  # 限制合理范围

    def inject_neighbor_crosstalk(self, B_signal, neighbors):
        """注入邻位车辆串扰（完整物理模型）

        地下车库标准车位几何：
        - 横向邻位：左1(2.5m), 左2(5.0m), 右1(2.5m), 右2(5.0m)
        - 纵向邻位：前对(5.5m), 后对(5.5m)
        """
        B_out = B_signal.copy()

        # 完整邻位位置映射（本位在原点，车辆沿x轴停放）
        neighbor_positions = {
            # 横向邻位（y轴方向）
            'left_1': np.array([0.0, -2.5, 0.0]),
            'left_2': np.array([0.0, -5.0, 0.0]),
            'right_1': np.array([0.0, 2.5, 0.0]),
            'right_2': np.array([0.0, 5.0, 0.0]),
            # 纵向邻位（x轴方向，对面行）
            'front': np.array([-5.5, 0.0, 0.0]),
            'back': np.array([5.5, 0.0, 0.0]),
        }

        # 按距离快速查找（兼容旧接口）
        distance_to_position = {
            2.5: np.array([0.0, 2.5, 0.0]),  # 默认右侧邻位
            5.0: np.array([0.0, 5.0, 0.0]),  # 默认右二邻位
            5.5: np.array([-5.5, 0.0, 0.0]),  # 默认前面对位
        }

        for nb in neighbors:
            sim_temp = GeomagSimulator(B_earth=np.zeros(3), seed=self.rng.randint(1, 10000))
            B_nb = sim_temp.inject_self_occupancy(np.zeros_like(B_out), nb['s'])

            dist = nb['r']
            scale = (0.5 / dist) ** 3

            # 计算精确投影系数
            r_vec = distance_to_position.get(dist, np.array([dist, 0.0, 0.0]))
            proj = self._compute_dipole_projection(r_vec)

            B_out += B_nb * scale * proj

        return B_out

    def inject_em_interference(self, B_signal, A_total):
        """
        注入充电桩 EMI（含谐波混叠和粉红噪声）

        - 能同时影响空闲和占用时段
        - 添加随机脉冲干扰，造成随机分类错误
        - 通过提高A_total的有效作用强度，让干扰足以淹没车辆信号

        参数:
            B_signal: 原始磁场信号 (N, 3)
            A_total: EMI总幅值 (μT)，建议20-40

        返回:
            添加EMI干扰后的磁场信号 (N, 3)
        """
        N = B_signal.shape[0]
        fs = self.fs

        weights = np.array(EM_HARMONIC_WEIGHTS)
        A_k = A_total * weights / weights.sum()

        alias_freqs, phases = [], []
        for k, hf in enumerate(EM_HARMONIC_FREQ, start=1):
            fk = hf * 1000
            f_alias = np.abs(np.mod(fk + fs / 2, fs) - fs / 2)
            alias_freqs.append(f_alias)
            phases.append(self.rng.rand() * 2 * np.pi)

        theta = np.arccos(self.rng.uniform(-1, 1))
        phi = self.rng.uniform(0, 2 * np.pi)
        e_int = np.array([np.sin(theta) * np.cos(phi),
                          np.sin(theta) * np.sin(phi),
                          np.cos(theta)])

        pink = np.cumsum(self.rng.randn(N)) * 0.03
        pink = pink / np.std(pink) * (0.5 * A_total)

        t = np.arange(N) / fs
        harmonic_sum = np.zeros(N)
        for f_alias, A, phase in zip(alias_freqs, A_k, phases):
            harmonic_sum += A * np.sin(2 * np.pi * f_alias * t + phase)

        interference = (harmonic_sum + pink)
        B_out = B_signal + np.outer(interference, e_int)

        spike_rate = 0.15
        n_spikes = int(N * spike_rate)
        spike_indices = self.rng.choice(N, n_spikes, replace=False)

        for idx in spike_indices:
            spike_magnitude = self.rng.uniform(A_total * 0.3, A_total * 0.8)
            spike_direction = np.random.randn(3)
            spike_direction /= np.linalg.norm(spike_direction)
            B_out[idx] += spike_magnitude * spike_direction

        return B_out

    def inject_drift(self, B_signal, severe=1):
        """强基线漂移（可叠加阶跃）

        算法5.5.4-C：
        - 基准σ_drift = 0.03 μT/√s
        - 强漂移模式通过倍率提升σ值
        # - 每5±2分钟插入一次阶跃事件

        参数：
            severe: int或float，漂移倍率
                - 1: σ = 0.03 × 1 = 0.03 μT/√s (正常)
                - 3: σ = 0.03 × 3 = 0.09 μT/√s
                - 10: σ = 0.03 × 10 = 0.30 μT/√s
                - 30: σ = 0.03 × 30 = 0.90 μT/√s
                - 100: σ = 0.03 × 100 = 3.00 μT/√s (极强)
        """
        N = B_signal.shape[0]
        dt = 1 / self.fs

        # 基准σ_drift（与config.py中SIGMA_DRIFT保持一致）
        sigma_base = self.sigma_drift
        sigma = sigma_base * severe

        B_out = B_signal.copy()
        drift = np.zeros(3)
        for i in range(N):
            drift += self.rng.normal(0, sigma * np.sqrt(dt), 3)
            B_out[i] += drift

        # 随机阶跃事件：每5±2分钟
        step_interval = int(5 * 60 * self.fs)  # 5分钟基准
        jitter = int(2 * 60 * self.fs)  # ±2分钟扰动
        for start in range(0, N, step_interval + self.rng.randint(-jitter, jitter + 1)):
            if start >= N:
                break
            step_dir = self.rng.randn(3)
            step_dir /= np.linalg.norm(step_dir)
            step_mag = self.rng.uniform(-2.0, 2.0)
            B_out[start:] += step_mag * step_dir

        return B_out

    def inject_packet_loss(self, B_signal, p):
        """丢包，返回 (信号, 掩码)"""
        mask = self.rng.rand(B_signal.shape[0]) >= p
        B_out = B_signal.copy()
        B_out[~mask] = np.nan
        return B_out, mask

    def inject_sensor_fault(self, B_signal, fault_type):
        """传感器故障：garbage / stuck / offline"""
        N = B_signal.shape[0]
        if fault_type == 'garbage':
            noise = self.rng.normal(0, 50.0, (N, 3))
            return self.B_earth + noise, np.ones(N, dtype=bool)
        elif fault_type == 'stuck':
            idx = self.rng.randint(0, N)
            stuck_val = B_signal[idx]
            return np.tile(stuck_val, (N, 1)), np.ones(N, dtype=bool)
        elif fault_type == 'offline':
            return B_signal, np.zeros(N, dtype=bool)
        return B_signal, np.ones(N, dtype=bool)


def self_test_one_week():
    sim = GeomagSimulator()
    s = generate_event_sequence(duration_hours=24 * 7, seed=415411)
    B = sim.generate_baseline(T_seconds=7 * 24 * 3600)
    B = sim.inject_self_occupancy(B, s)
    base = B.__copy__()

    # B=sim.inject_drift(B,30)
    # B=sim.inject_em_interference(B, 20)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    # 大幅降采样
    step = 60 * 10 * 5  # 每60秒取1个点（10Hz下为600个采样点）
    t = np.arange(0, len(B), step) / (FS * 3600)  # 转换为小时

    axes[0].plot(t, B[::step, 2], linewidth=0.5, color='b', label='B_z')
    axes[0].plot(t, base[::step, 2], linewidth=0.5, color='r', label='B_z_base')
    axes[0].set_title('B_z over one week')
    axes[0].set_xlabel('Time (hours)')
    axes[0].set_ylabel('Magnetic Field (μT)')
    axes[0].set_xlim(0, 24 * 7)
    axes[0].set_xticks(np.arange(0, 24 * 7))

    axes[1].plot(t, s[::step], linewidth=0.5)
    axes[1].set_title('Occupancy')
    axes[1].set_xlabel('Time (hours)')
    axes[1].set_ylabel('Status')
    axes[1].set_xlim(0, 24 * 7)
    axes[1].set_xticks(np.arange(0, 24 * 7))

    plt.tight_layout()
    plt.savefig('results/sanity_geomag_oneweek.png', dpi=150, bbox_inches='tight')
    plt.close()

    occ_rate = s.mean()
    n_events = len(np.where(np.diff(s, prepend=0) == 1)[0])
    print(f"✓ 一周测试完成")
    print(f"  采样点数: {len(B):,}")
    print(f"  占用率: {occ_rate:.2%}")
    print(f"  停车事件数: {n_events}")


def test_baseline():
    sim = GeomagSimulator()
    B = sim.generate_baseline(T_seconds=24 * 3600)
    import matplotlib.pyplot as plt
    plt.plot(B[:, 2])
    # plt.legend()
    plt.savefig('results/sanity_geomag_baseline.png', bbox_inches='tight')


if __name__ == '__main__':
    self_test_one_week()
    # test_baseline()
