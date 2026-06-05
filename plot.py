# plot.py

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import RESULTS_DIR

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def h1_plot():
    # 图 1：退化曲线（2行3列布局）
    h1_path = os.path.join(RESULTS_DIR, 'h1_results.csv')
    if os.path.exists(h1_path):
        df = pd.read_csv(h1_path)
        channels = {'cv': '视觉', 'us': '超声', 'mag': '地磁'}

        # 根据实际CSV数据动态提取存在的failure类型
        failures_list = {
            'cv': ['gamma', 'noise'],
            'us': ['multipath'],
            'mag': ['em', 'drift']
        }

        # 统计需要绘制的子图数量
        total_plots = sum(len(v) for v in failures_list.values())

        # 2行3列布局，下方两个图表居中
        fig = plt.figure(figsize=(7, 4.5))
        gs = fig.add_gridspec(2, 6, hspace=0.3, wspace=0.3)

        # 第一行：3个图表占据全部6列（每个占2列）
        # 第二行：2个图表居中（各占2列，从第2列开始）
        axes_positions = [
            gs[0, 0:2],  # 第1行第1个
            gs[0, 2:4],  # 第1行第2个
            gs[0, 4:6],  # 第1行第3个
            gs[1, 1:3],  # 第2行第1个（居中）
            gs[1, 3:5],  # 第2行第2个（居中）
        ]

        axes = [fig.add_subplot(pos) for pos in axes_positions]

        plot_idx = 0
        for ch in ['cv', 'us', 'mag']:
            ch_df = df[df['channel'] == ch]
            for fail in failures_list.get(ch, []):
                if plot_idx >= len(axes):
                    break

                ax = axes[plot_idx]
                sub = ch_df[ch_df['failure'] == fail].sort_values('severity')

                # 绘制退化曲线
                ax.plot(sub['severity'], sub['accuracy'], 'o-', linewidth=1.5, markersize=4, color='#2E86AB')
                ax.axhline(0.85, color='red', linestyle='--', linewidth=1.0, alpha=0.7, label='Threshold=0.85')

                # 设置标题和标签
                ax.set_title(f'{channels[ch]}-{fail}', fontsize=9)
                ax.set_xlabel('Severity', fontsize=8)
                ax.set_ylabel('Accuracy', fontsize=8)
                ax.set_ylim(0, 1.05)

                # 动态设置x轴刻度
                severities = sub['severity'].values
                if len(severities) > 0:
                    ax.set_xticks(sorted(set(severities)))

                ax.grid(True, alpha=0.3, linestyle=':')
                ax.tick_params(labelsize=7)

                # 只在第一个子图显示图例
                if plot_idx == 0:
                    ax.legend(fontsize=6, loc='lower left')

                plot_idx += 1

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'fig1_degradation_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("H1退化曲线已保存 (2×3布局，下方居中)")
    else:
        print("未找到 h1_results.csv，跳过图1")


def h2_plot_v2():
    """
    H2 实验结果可视化（多失效程度版本）

    图1（单图）：
      - 不同退化程度下各通道准确率对比折线图（基于 h2_results.csv）

    图2（5个子图）：
      - 每个子图对应一个失效等级（sev0~sev4）
      - 展示该等级下各通道的滚动窗口准确率时序
    """

    # ==================== 图 1：准确率 vs 失效程度 ====================
    h2_results_path = os.path.join(RESULTS_DIR, 'h2_results.csv')

    if not os.path.exists(h2_results_path):
        print("未找到 h2_results.csv，跳过图1")
        return

    h2_results = pd.read_csv(h2_results_path)

    fig1, ax1 = plt.subplots(figsize=(7, 5))

    severity_levels = h2_results['severity_level'].values
    acc_v_mean = h2_results['acc_v_mean'].values
    acc_m_mean = h2_results['acc_m_mean'].values
    acc_u_mean = h2_results['acc_u_mean'].values
    acc_fused_mean = h2_results['acc_fused_mean'].values

    x_pos = np.arange(len(severity_levels))

    ax1.plot(x_pos, acc_v_mean, marker='o', linewidth=2.5, markersize=8,
             label='Visual', color='#2E86AB', alpha=0.9)
    ax1.plot(x_pos, acc_m_mean, marker='s', linewidth=2.5, markersize=8,
             label='Magnetic', color='#A23B72', alpha=0.9)
    ax1.plot(x_pos, acc_u_mean, marker='^', linewidth=2.5, markersize=8,
             label='Ultrasonic', color='#F18F01', alpha=0.9)
    ax1.plot(x_pos, acc_fused_mean, marker='D', linewidth=2.5, markersize=8,
             label='Fused', color='#000000', alpha=0.9)

    ax1.set_xlabel('Severity Level', fontsize=11, labelpad=8)
    ax1.set_ylabel('Average Accuracy', fontsize=11, labelpad=8)
    ax1.set_title('Accuracy vs. Degradation Severity', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'Sev {int(s)}' for s in severity_levels])
    ax1.set_ylim(0, 1.15)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=4)
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig1_h2_accuracy_severity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("图1：H2 准确率-失效程度折线图已保存")

    # ==================== 图 2：5个失效等级的时序图 ====================
    fig2, axes = plt.subplots(5, 1, figsize=(12, 20), sharex=True)

    severity_files = [
        ('sev0', 'Severity 0 (No Degradation)'),
        ('sev1', 'Severity 1 (Light)'),
        ('sev2', 'Severity 2 (Moderate)'),
        ('sev3', 'Severity 3 (Severe)'),
        ('sev4', 'Severity 4 (Critical)')
    ]

    colors = {
        'Visual': '#2E86AB',
        'Magnetic': '#A23B72',
        'Ultrasonic': '#F18F01',
        'Fused': '#000000'
    }

    for idx, (sev_name, sev_title) in enumerate(severity_files):
        csv_path = os.path.join(RESULTS_DIR, f'h2_{sev_name}_window_acc.csv')

        if not os.path.exists(csv_path):
            print(f"  警告：未找到 {csv_path}，跳过子图 {idx}")
            continue

        df = pd.read_csv(csv_path)
        ax = axes[idx]

        t_minutes = np.arange(len(df)) * 10
        t_hours = t_minutes / 60.0

        sample_rate = 3
        t_sampled = t_hours[::sample_rate]
        acc_v_sampled = df['acc_v'].values[::sample_rate]
        acc_m_sampled = df['acc_m'].values[::sample_rate]
        acc_u_sampled = df['acc_u'].values[::sample_rate]
        # acc_fused_sampled = df['acc_fused'].values[::sample_rate]

        ax.plot(t_sampled, acc_v_sampled, label='Visual', alpha=0.7,
                linewidth=1.5, color=colors['Visual'])
        ax.plot(t_sampled, acc_m_sampled, label='Magnetic', alpha=0.7,
                linewidth=1.5, color=colors['Magnetic'])
        ax.plot(t_sampled, acc_u_sampled, label='Ultrasonic', alpha=0.7,
                linewidth=1.5, color=colors['Ultrasonic'])
        # ax.plot(t_sampled, acc_fused_sampled, label='Fused', linewidth=2.0,
        #         alpha=0.8, color=colors['Fused'])

        ax.set_ylabel('Accuracy', fontsize=10)
        ax.set_title(sev_title, fontsize=11, fontweight='bold', loc='left', pad=5)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        ax.set_ylim(-0.05, 1.05)

        if idx == len(severity_files) - 1:
            ax.set_xlabel('Time (hours)', fontsize=11)
        else:
            ax.tick_params(labelbottom=False)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig2.legend(handles, labels, loc='lower center', ncol=4, fontsize=10,
                framealpha=0.9, borderaxespad=1.0)

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig(os.path.join(RESULTS_DIR, 'fig2_h2_severity_timeseries.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("图2：5个失效等级时序图已保存")


def h2_plot_scenes():
    """
    绘制两张图：
    图1：7个场景（单/双/全失效）下四通道平均准确率折线图
    图2：7个场景各自窗口级准确率时序图（4通道）
    """
    summary_path = os.path.join(RESULTS_DIR, 'h2_scene_summary.csv')
    if not os.path.exists(summary_path):
        print("未找到 h2_scene_summary.csv，跳过 H2 场景绘图")
        return

    # 读取场景汇总
    summary_df = pd.read_csv(summary_path)
    scenes = summary_df['scene'].tolist()
    # 定义显示顺序
    ordered_scenes = ['v', 'm', 'u', 'vm', 'vu', 'mu', 'vmu']
    # 确保数据按顺序排列
    summary_df['scene'] = pd.Categorical(summary_df['scene'], categories=ordered_scenes, ordered=True)
    summary_df = summary_df.sort_values('scene')

    # ==================== 图1：场景平均准确率折线图 ====================
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(summary_df))
    ax1.plot(x_pos, summary_df['acc_v_mean'], marker='o', linewidth=2.5, markersize=8,
             label='Visual', color='#2E86AB')
    ax1.plot(x_pos, summary_df['acc_m_mean'], marker='s', linewidth=2.5, markersize=8,
             label='Magnetic', color='#A23B72')
    ax1.plot(x_pos, summary_df['acc_u_mean'], marker='^', linewidth=2.5, markersize=8,
             label='Ultrasonic', color='#F18F01')
    ax1.plot(x_pos, summary_df['acc_fused_mean'], marker='D', linewidth=2.5, markersize=8,
             label='Fused (D-S)', color='#000000')
    # 在数据点旁边添加数值标签
    for method in ['acc_v_mean', 'acc_m_mean', 'acc_u_mean', 'acc_fused_mean']:
        for i, value in enumerate(summary_df[method]):
            ax1.text(i, value + 0.015, f'{value:.3f}', ha='center', va='bottom', fontsize=7)

    # 标注虚线分隔三类场景
    ax1.axvline(x=2.5, color='grey', linestyle='--', alpha=0.5)
    ax1.axvline(x=5.5, color='grey', linestyle='--', alpha=0.5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(summary_df['scene'])
    ax1.set_xlabel('Failure Scenario', fontsize=12)
    ax1.set_ylabel('Average Accuracy', fontsize=12)
    ax1.set_title('Average Accuracy under Different Failure Scenarios', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig_h2_scene_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("图1：场景平均准确率折线图已保存")

    # ==================== 图2：7个场景窗口级时序图 ====================
    fig2, axes = plt.subplots(len(ordered_scenes), 1, figsize=(14, 3 * len(ordered_scenes)), sharex=True)
    colors = {'Visual': '#2E86AB', 'Magnetic': '#A23B72', 'Ultrasonic': '#F18F01', 'Fused': '#000000'}
    for idx, scene in enumerate(ordered_scenes):
        csv_file = os.path.join(RESULTS_DIR, f'h2_scene_{scene}_window.csv')
        if not os.path.exists(csv_file):
            print(f"  警告：未找到 {csv_file}，跳过场景 {scene}")
            continue
        df = pd.read_csv(csv_file)
        ax = axes[idx]
        t_min = np.arange(len(df)) * 10  # 分钟
        t_hour = t_min / 60.0
        # 降采样：每3个点取1个
        sample_rate = 3
        t_sampled = t_hour[::sample_rate]
        ax.plot(t_sampled, df['acc_v'].values[::sample_rate], label='Visual', alpha=0.7,
                linewidth=1.5, color=colors['Visual'])
        ax.plot(t_sampled, df['acc_m'].values[::sample_rate], label='Magnetic', alpha=0.7,
                linewidth=1.5, color=colors['Magnetic'])
        ax.plot(t_sampled, df['acc_u'].values[::sample_rate], label='Ultrasonic', alpha=0.7,
                linewidth=1.5, color=colors['Ultrasonic'])
        ax.plot(t_sampled, df['acc_fused'].values[::sample_rate], label='Fused', alpha=0.9,
                linewidth=2.0, color=colors['Fused'])
        ax.set_ylabel('Accuracy', fontsize=10)
        ax.set_title(f'Scenario: {scene}', fontsize=11, loc='left', pad=3)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_ylim(-0.05, 1.05)
        if idx == len(ordered_scenes) - 1:
            ax.set_xlabel('Time (hours)', fontsize=11)
        else:
            ax.tick_params(labelbottom=False)
    # 统一图例
    handles, labels = axes[-1].get_legend_handles_labels()
    fig2.legend(handles, labels, loc='lower center', ncol=4, fontsize=10, framealpha=0.9)
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.savefig(os.path.join(RESULTS_DIR, 'fig_h2_scene_timeseries.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("图2：7个场景窗口时序图已保存")


def h3_plot_scenes():
    """
    绘制 H3 实验结果（融合算法对比）的两张图：
    图1：3个场景（vm/vu/mu）下各融合方法的平均准确率对比折线图
    图2：3个场景各自窗口级准确率时序图（单通道 + 三种融合方法）
    """
    summary_path = os.path.join(RESULTS_DIR, 'h3_results.csv')
    if not os.path.exists(summary_path):
        print("未找到 h3_results.csv，跳过 H3 场景绘图")
        return

    # 读取场景汇总
    summary_df = pd.read_csv(summary_path)

    # 定义显示顺序
    ordered_scenes = ['v', 'm', 'u', 'vm', 'vu', 'mu', 'vmu']
    summary_df['scene'] = pd.Categorical(summary_df['scene'], categories=ordered_scenes, ordered=True)
    summary_df = summary_df.sort_values('scene')

    # ==================== 图1：场景平均准确率对比折线图 ====================
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(summary_df))

    # 定义颜色
    colors = {
        'Visual': '#2E86AB',
        'Magnetic': '#A23B72',
        'Ultrasonic': '#F18F01',
        'Majority Vote': '#06A77D',
        'DS-Equal': '#D62828',
        'DS-Static': '#6A4C93'
    }

    # 绘制各方法的折线图
    ax1.plot(x_pos, summary_df['acc_v_mean'], marker='o', markersize=8, linewidth=2.0,
             label='Visual', color=colors['Visual'], linestyle='--')
    ax1.plot(x_pos, summary_df['acc_m_mean'], marker='s', markersize=8, linewidth=2.0,
             label='Magnetic', color=colors['Magnetic'], linestyle='--')
    ax1.plot(x_pos, summary_df['acc_u_mean'], marker='^', markersize=8, linewidth=2.0,
             label='Ultrasonic', color=colors['Ultrasonic'], linestyle='--')
    ax1.plot(x_pos, summary_df['acc_mv_mean'], marker='d', markersize=8, linewidth=2.0,
             label='Majority Vote', color=colors['Majority Vote'])
    ax1.plot(x_pos, summary_df['acc_ds_eq_mean'], marker='p', markersize=8, linewidth=2.0,
             label='DS-Equal', color=colors['DS-Equal'])
    ax1.plot(x_pos, summary_df['acc_ds_static_mean'], marker='*', markersize=10, linewidth=2.0,
             label='DS-Static', color=colors['DS-Static'])

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(summary_df['scene'], fontsize=12)
    ax1.set_xlabel('Failure Scenario', fontsize=12)
    ax1.set_ylabel('Average Accuracy', fontsize=12)
    ax1.set_title('Average Accuracy Comparison under Different Fusion Methods', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_ylim(0, 1.05)

    # 在数据点旁边添加数值标签
    for method in ['acc_v_mean', 'acc_m_mean', 'acc_u_mean', 'acc_mv_mean', 'acc_ds_eq_mean', 'acc_ds_static_mean']:
        for i, value in enumerate(summary_df[method]):
            ax1.text(i, value + 0.015, f'{value:.3f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig_h3_scene_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("图1：H3 场景平均准确率对比图已保存")

    # ==================== 图2：3个场景窗口级时序图 ====================
    fig2, axes = plt.subplots(len(ordered_scenes), 1, figsize=(14, 3 * len(ordered_scenes)), sharex=True)

    colors_ts = {
        'Visual': '#2E86AB',
        'Magnetic': '#A23B72',
        'Ultrasonic': '#F18F01',
        'Majority Vote': '#06A77D',
        'DS-Equal': '#D62828',
        'DS-Static': '#6A4C93'
    }

    for idx, scene in enumerate(ordered_scenes):
        csv_file = os.path.join(RESULTS_DIR, f'h3_scene_{scene}_window.csv')
        if not os.path.exists(csv_file):
            print(f"  警告：未找到 {csv_file}，跳过场景 {scene}")
            continue
        df = pd.read_csv(csv_file)
        ax = axes[idx]
        t_min = np.arange(len(df)) * 10  # 分钟
        t_hour = t_min / 60.0

        # 降采样：每3个点取1个
        sample_rate = 3
        t_sampled = t_hour[::sample_rate]

        ax.plot(t_sampled, df['acc_v'].values[::sample_rate], label='Visual', alpha=0.7,
                linewidth=1.5, color=colors_ts['Visual'])
        ax.plot(t_sampled, df['acc_m'].values[::sample_rate], label='Magnetic', alpha=0.7,
                linewidth=1.5, color=colors_ts['Magnetic'])
        ax.plot(t_sampled, df['acc_u'].values[::sample_rate], label='Ultrasonic', alpha=0.7,
                linewidth=1.5, color=colors_ts['Ultrasonic'])
        ax.plot(t_sampled, df['acc_mv'].values[::sample_rate], label='Majority Vote', alpha=0.9,
                linewidth=2.0, color=colors_ts['Majority Vote'])
        ax.plot(t_sampled, df['acc_ds_eq'].values[::sample_rate], label='DS-Equal', alpha=0.9,
                linewidth=2.0, color=colors_ts['DS-Equal'])
        ax.plot(t_sampled, df['acc_ds_static'].values[::sample_rate], label='DS-Static', alpha=0.9,
                linewidth=2.0, color=colors_ts['DS-Static'])

        ax.set_ylabel('Accuracy', fontsize=10)
        ax.set_title(f'Scenario: {scene}', fontsize=11, loc='left', pad=3)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_ylim(-0.05, 1.05)

        if idx == len(ordered_scenes) - 1:
            ax.set_xlabel('Time (hours)', fontsize=11)
        else:
            ax.tick_params(labelbottom=False)

    # 统一图例
    handles, labels = axes[-1].get_legend_handles_labels()
    fig2.legend(handles, labels, loc='lower center', ncol=3, fontsize=9, framealpha=0.9)
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.savefig(os.path.join(RESULTS_DIR, 'fig_h3_scene_timeseries.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("图2：H3 场景窗口时序图已保存")


def h4_plot_scenes():
    """
    绘制 H4 实验结果，每个场景下不同严重程度的准确率折线图。
    """
    h4_path = os.path.join(RESULTS_DIR, 'h4_results.csv')
    if not os.path.exists(h4_path):
        print("未找到 h4_results.csv，跳过 H4 场景绘图")
        return

    df = pd.read_csv(h4_path)

    single_channel_scenes = ['v', 'm', 'u']
    multi_channel_scenes = ['vm', 'vu', 'mu', 'vmu']

    methods = {
        'Visual': ('acc_v_mean', '#2E86AB', '--'),
        'Magnetic': ('acc_m_mean', '#A23B72', '--'),
        'Ultrasonic': ('acc_u_mean', '#F18F01', '--'),
        'MV': ('acc_mv_mean', '#06A77D', '-'),
        'DS-Equal': ('acc_ds_eq_mean', '#D62828', '-'),
        'DS-Static': ('acc_ds_static_mean', '#6A4C93', '-'),
        'Ours': ('acc_ours_mean', '#000000', '-'),
    }

    def plot_scene_group(scenes, filename, title_prefix):
        n_scenes = len(scenes)
        fig, axes = plt.subplots(1, n_scenes, figsize=(5.5 * n_scenes, 3.5), sharey=False)
        if n_scenes == 1:
            axes = [axes]

        for idx, scene in enumerate(scenes):
            scene_df = df[df['scene'] == scene].sort_values('level')
            if scene_df.empty:
                continue

            ax = axes[idx]
            levels = scene_df['level'].values

            for method_name, (col_name, color, linestyle) in methods.items():
                acc_vals = scene_df[col_name].values
                if method_name == 'Ours':
                    ax.plot(levels, acc_vals, marker='o', markersize=4,
                            linewidth=2.2, linestyle=linestyle, color=color, label=method_name)
                else :
                    ax.plot(levels, acc_vals, marker='o', markersize=4,
                            linewidth=1.8, linestyle=linestyle, color=color, label=method_name)
                # for x, y in zip(levels, acc_vals):
                #     if y > 0.5:
                #         ax.text(x, y + 0.02, f'{y:.3f}', ha='center', va='bottom',
                #                 fontsize=11, color=color)

            ax.set_title(f'Degeneration Scene {scene.upper()}', fontsize=20, fontweight='bold')
            ax.set_xticks(levels)
            # 添加y轴坐标
            ax.set_ylabel('Accuracy', fontsize=20)
            ax.set_ylim(0.5, 1.08)
            ax.grid(True, alpha=0.3, linestyle=':')

            ax.tick_params(axis='both', which='major', labelsize=16)

        axes[0].set_ylabel('Average Accuracy', fontsize=20)
        # fig.suptitle(title_prefix, fontsize=26, fontweight='bold', y=1.02)

        # 统一的图例放在所有子图下方
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=7, fontsize=18, framealpha=0.9,
                   bbox_to_anchor=(0.5, -0.05))
        plt.subplots_adjust(bottom=0.2, wspace=0.25, hspace=0.3)
        plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{title_prefix} 已保存: {filename}")

    plot_scene_group(single_channel_scenes, 'fig_h4_single_channel.png', 'Single-Channel Degradation Performance')
    plot_scene_group(multi_channel_scenes, 'fig_h4_multi_channel.png', 'Multi-Channel Degradation Performance')
    print("H4 全部场景绘图完成")


def h4_plot_window_timeseries():
    """
    绘制 H4 实验窗口级时序图
    - 对每个场景×退化等级组合，绘制窗口级准确率时序
    - 展示各通道和融合方法的动态表现
    - 降采样以提高可读性
    """
    # scenes = ['v', 'm', 'u', 'vm', 'vu', 'mu', 'vmu']
    scenes = ['vu']
    # levels = [0, 1, 2, 3, 4]
    levels = [2]

    colors_ts = {
        'Visual': '#2E86AB',
        'Magnetic': '#A23B72',
        'Ultrasonic': '#F18F01',
        'MV': '#06A77D',
        'DS-Equal': '#D62828',
        'DS-Static': '#6A4C93',
        'Ours': '#000000'
    }

    plot_count = 0

    for scene in scenes:
        for level in levels:
            csv_file = os.path.join(RESULTS_DIR, f'h4_scene_{scene}_lev{level}_window.csv')
            if not os.path.exists(csv_file):
                print(f"  警告：未找到 {csv_file}，跳过场景 {scene} 等级 {level}")
                continue

            df = pd.read_csv(csv_file)

            fig, ax = plt.subplots(figsize=(14, 5))
            t_min = np.arange(len(df)) * 10  # 分钟
            t_hour = t_min / 60.0

            # 降采样：每3个点取1个
            sample_rate = 3
            t_sampled = t_hour[::sample_rate]

            # 绘制各通道和融合方法
            ax.plot(t_sampled, df['acc_v'].values[::sample_rate], label='Visual', alpha=0.7,
                    linewidth=1.5, color=colors_ts['Visual'])
            ax.plot(t_sampled, df['acc_m'].values[::sample_rate], label='Magnetic', alpha=0.7,
                    linewidth=1.5, color=colors_ts['Magnetic'])
            ax.plot(t_sampled, df['acc_u'].values[::sample_rate], label='Ultrasonic', alpha=0.7,
                    linewidth=1.5, color=colors_ts['Ultrasonic'])
            ax.plot(t_sampled, df['acc_mv'].values[::sample_rate], label='MV', alpha=0.9,
                    linewidth=2.0, color=colors_ts['MV'])
            ax.plot(t_sampled, df['acc_ds_eq'].values[::sample_rate], label='DS-Equal', alpha=0.9,
                    linewidth=2.0, color=colors_ts['DS-Equal'])
            ax.plot(t_sampled, df['acc_ds_static'].values[::sample_rate], label='DS-Static', alpha=0.9,
                    linewidth=2.0, color=colors_ts['DS-Static'])
            ax.plot(t_sampled, df['acc_ours'].values[::sample_rate], label='Ours', alpha=0.9,
                    linewidth=2.0, color=colors_ts['Ours'])

            ax.set_xlabel('Time (hours)', fontsize=22)
            ax.set_ylabel('Accuracy', fontsize=22)
            # ax.set_title(f'Scene: {scene} | Degradation Level: {level}',
            #              fontsize=24, fontweight='bold', loc='left', pad=8)
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.set_ylim(-0.05, 1.05)
            ax.legend(loc='lower left', fontsize=18, ncol=4)
            ax.tick_params(axis='both', which='major', labelsize=16)

            plt.tight_layout()
            save_path = os.path.join(RESULTS_DIR, f'fig_h4_{scene}_lev{level}_timeseries.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            plot_count += 1
            print(f"  已保存: {save_path}")

    print(f"H4 窗口级时序图绘制完成，共 {plot_count} 张图")


def ablation_plot():
    csv_path = os.path.join(RESULTS_DIR, 'ablation_results.csv')
    if not os.path.exists(csv_path):
        print("未找到 ablation_results.csv")
        return
    df = pd.read_csv(csv_path)

    variants = ['Full', '-Weight', '-Conflict', '-Adapt']
    colors = {'Full': '#000000', '-Weight': '#06A77D', '-Conflict': '#D62828', '-Adapt': '#6A4C93'}

    # 单通道退化场景
    single_scenes = ['v', 'm', 'u']
    # 多通道退化场景
    multi_scenes = ['vm',  'mu', 'vu', 'vmu']

    # 绘制单通道退化场景
    _plot_ablation_scene(df, single_scenes, variants, colors,
                         'fig_ablation_single_channel.png',
                         'Ablation Study: Single Channel Degradation')

    # 绘制多通道退化场景
    _plot_ablation_scene(df, multi_scenes, variants, colors,
                         'fig_ablation_multi_channel.png',
                         'Ablation Study: Multi-Channel Degradation')

    print("消融实验对比图已保存至 fig_ablation_single_channel.png 和 fig_ablation_multi_channel.png")


def _plot_ablation_scene(df, scenes, variants, colors, filename, title):
    """绘制消融实验场景图"""
    n_scenes = len(scenes)

    if n_scenes <= 4:
        # 如果场景数<=4，使用单排布局
        fig, axes = plt.subplots(1, n_scenes, figsize=(5.5 * n_scenes, 3.5), sharey=False)
        if n_scenes == 1:
            axes = [axes]
        axes_list = axes
    else:
        # 如果场景数>4，使用双排布局
        fig = plt.figure(figsize=(22, 7))
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
        axes_top = [fig.add_subplot(gs[0, i]) for i in range(4)]
        axes_bottom = [fig.add_subplot(gs[1, i]) for i in range(1, 4)]
        axes_list = axes_top + axes_bottom

    for i, scene in enumerate(scenes):
        ax = axes_list[i]
        scene_df = df[df['scene'] == scene].sort_values('level')
        for var in variants:
            vals = scene_df[var].values
            lv = scene_df['level'].values
            ax.plot(lv, vals, marker='o', markersize=5, linewidth=1.8,
                    label=var, color=colors[var])
        # 设置标题和坐标轴
        ax.set_title(f'Degradation Scene {scene.upper()}', fontsize=20, fontweight='bold')
        ax.set_xlabel('Degradat Level', fontsize=18)
        ax.set_xticks([0, 1, 2, 3, 4])

        ax.set_ylim(0, 1.08)
        ax.set_ylabel('Accuracy', fontsize=20)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        # 设置坐标轴字体大小
        ax.tick_params(axis='both', which='major', labelsize=16)
        # 添加网格
        ax.grid(True, alpha=0.5, linestyle=':', color='black')
    # 添加图例
    handles, labels = axes_list[-1].get_legend_handles_labels()
    if len(scenes) <= 4:
        fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=18,
                   framealpha=0.9, bbox_to_anchor=(0.5, -0.15))
        plt.subplots_adjust(bottom=0.25, wspace=0.3)
    else:
        fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=18,
                   framealpha=0.9, bbox_to_anchor=(0.5, -0.05))
        plt.subplots_adjust(bottom=0.15, wspace=0.3, hspace=0.3)

    # plt.suptitle(title, fontsize=26, fontweight='bold')
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # h1_plot()
    # h2_plot_v2()
    # h2_plot_scenes()
    # h3_plot_scenes()
    # h4_plot_scenes()
    # h4_plot_window_timeseries()
    ablation_plot()
