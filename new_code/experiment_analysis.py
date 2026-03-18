import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# 引入核心组件
from basic_config import CONFIG, DRIVER_CONFIG
from ride_hailing_env import RideHailingEnv
from shared_ppo import SharedPPOAgent  # 确保 shared_ppo.py 在同一目录下
from generate_simulator import PassengerSimulator


# 组装参数
platform_params = {
        'surge': lambda t, no, nd, sd: 1.0 + 1.2 * np.maximum(0, sd - 0.8) ** 1.2,
        'subsidy': lambda t, no, nd, sd: 3.0 * (np.exp(-50 * (t - 0.30) ** 2) + np.exp(-40 * (t - 0.75) ** 2)) * np.maximum(0, sd - 0.5)
}

# ==================== 2. 实验核心逻辑 ====================
def run_comparison_experiment(simulator_path, model_path='compare_down_alg/ppo_agent_final.pth'):
    # 实验设置：不同司机规模
    driver_scales = [500, 1000, 1500, 2000]

    # 结果容器
    results = {
        'scales': driver_scales,
        'profits': [],
        'ft_incomes': [],  # 全职司机收入
        'pt_incomes': [],  # 兼职司机收入
        'ft_efficiency': [],  # 全职司机效率 (时薪)
        'pt_efficiency': [],  # 兼职司机效率 (时薪)
        'service_rates': []  # 订单响应率
    }

    # 预创建热力图目录
    os.makedirs('heatmaps', exist_ok=True)

    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"Warning: Model file '{model_path}' not found. Using random initialization!")
        model_exists = False
    else:
        print(f"Loading trained model from {model_path}...")
        model_exists = True

    # --- 开始遍历实验 ---
    for n_drivers in driver_scales:
        print(f"\n>>> Running Experiment with N_DRIVERS = {n_drivers}...")

        # 1. 动态修改全局配置
        CONFIG['N_DRIVERS'] = n_drivers

        # 2. 初始化环境
        env = RideHailingEnv(simulator_path)

        # 3. 初始化 Agent 并加载权重
        agent = SharedPPOAgent(CONFIG['STATE_DIM'], CONFIG['ACTION_DIM'], **CONFIG)
        if model_exists:
            agent.policy.load_state_dict(torch.load(model_path))
            agent.policy.eval()  # 切换到评估模式

        # 4. 重置环境
        state, action_mask = env.reset()
        done = False

        # 掩码辅助
        ft_mask = env.driver_type == 1
        pt_mask = env.driver_type == 0

        # --- 定义用于 KM 匹配的 Critic 预估器 (Batch版) ---
        def critic_estimator_batch(d_ids, dest_idxs):
            if not d_ids: return np.array([])
            # 批量获取 Critic 所需的伪状态
            proxy_states = [env.get_proxy_state(d, dest) for d, dest in zip(d_ids, dest_idxs)]
            state_tensor = torch.FloatTensor(np.array(proxy_states))

            with torch.no_grad():
                values = agent.policy.critic(state_tensor).squeeze(-1)
            return values.numpy()

        # --- 时间步循环 ---
        pbar = tqdm(total=CONFIG['TIME_STEPS_PER_DAY'])
        while not done:
            # 5. 模型决策 (Inference)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                mask_tensor = torch.BoolTensor(action_mask)
                # 使用 Policy 网络选择动作
                actions, _ = agent.policy.act(state_tensor, mask_tensor)
                actions = actions.numpy()

            # 6. 环境步进 (传入 Critic 预估器以支持二分图匹配)
            next_state, rewards, done, info = env.step(
                actions, platform_params,
                value_estimator=critic_estimator_batch
            )

            # 7. 每隔 20 步保存一次热力图 (仅针对 500 司机规模，避免图片过多)
            if n_drivers == 500 and env.time % 20 == 0:
                plot_spatial_heatmap(env, n_drivers, env.time)

            state = next_state
            action_mask = info['action_mask']
            pbar.update(1)
        pbar.close()

        # --- 收集本轮实验数据 ---
        total_profit = info['total_revenue']
        service_rate = info['total_served'] / (info['total_generated'] + 1e-6)

        avg_ft_income = np.mean(env.driver_daily_income[ft_mask]) if np.any(ft_mask) else 0
        avg_pt_income = np.mean(env.driver_daily_income[pt_mask]) if np.any(pt_mask) else 0

        # 计算每步效率 (避免除以0)
        ft_steps = env.driver_active_steps[ft_mask] + 1e-6
        pt_steps = env.driver_active_steps[pt_mask] + 1e-6
        avg_ft_eff = np.mean(env.driver_daily_income[ft_mask] / ft_steps) if np.any(ft_mask) else 0
        avg_pt_eff = np.mean(env.driver_daily_income[pt_mask] / pt_steps) if np.any(pt_mask) else 0

        results['profits'].append(total_profit)
        results['service_rates'].append(service_rate)
        results['ft_incomes'].append(avg_ft_income)
        results['pt_incomes'].append(avg_pt_income)
        results['ft_efficiency'].append(avg_ft_eff)
        results['pt_efficiency'].append(avg_pt_eff)

        print(f"Scale {n_drivers}: Profit={total_profit:.0f}, ServiceRate={service_rate:.2f}")

    # --- 绘制最终对比图 ---
    plot_results(results)


# ==================== 3. 绘图工具函数 ====================
def plot_spatial_heatmap(env, scale, time_step):
    """绘制: 订单分布 vs 全职司机分布 vs 兼职司机分布"""
    lats = env.hex_latlng_arr[:, 0]
    lngs = env.hex_latlng_arr[:, 1]

    # 统计数据
    order_counts = np.zeros(env.n_zones)
    for o in env.pending_orders:
        if not o['matched']: order_counts[o['origin_idx']] += 1

    ft_counts = np.zeros(env.n_zones)
    pt_counts = np.zeros(env.n_zones)

    # 快速统计司机位置
    online_mask = env.driver_status != 2
    ft_mask = (env.driver_type == 1) & online_mask
    pt_mask = (env.driver_type == 0) & online_mask

    np.add.at(ft_counts, env.driver_locations[ft_mask], 1)
    np.add.at(pt_counts, env.driver_locations[pt_mask], 1)

    # 绘图
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    def draw_scatter(ax, data, cmap, title):
        # 过滤掉 0 值以保持图面整洁
        mask = data > 0
        if not np.any(mask):
            ax.text(0.5, 0.5, "No Data", ha='center')
            return
        sc = ax.scatter(lngs[mask], lats[mask], c=data[mask],
                        cmap=cmap, s=data[mask] * 10 + 20, alpha=0.8, edgecolors='grey')
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    draw_scatter(axs[0], order_counts, 'Reds', f'Order Demand (T={time_step})')
    draw_scatter(axs[1], ft_counts, 'Blues', f'Full-Time Drivers (Scale={scale})')
    draw_scatter(axs[2], pt_counts, 'Greens', f'Part-Time Drivers (Scale={scale})')

    plt.suptitle(f'Spatial Distribution at Time Step {time_step}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'heatmaps/heatmap_scale_{scale}_t_{time_step:03d}.png')
    plt.close()


def plot_results(res):
    """绘制所有对比指标"""
    scales = res['scales']
    x = np.arange(len(scales))
    width = 0.35

    fig = plt.figure(figsize=(20, 10))

    # 1. 平台总利润
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(scales, res['profits'], 'o-', color='purple', linewidth=2)
    ax1.set_title('Total Platform Profit')
    ax1.set_xlabel('Driver Scale')
    ax1.grid(True, alpha=0.3)

    # 2. 订单响应率
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(scales, res['service_rates'], 's-', color='orange', linewidth=2)
    ax2.set_title('Service Completion Rate')
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)

    # 3. 司机日均收入对比
    ax3 = fig.add_subplot(2, 3, 4)
    ax3.bar(x - width / 2, res['ft_incomes'], width, label='Full-Time', color='royalblue')
    ax3.bar(x + width / 2, res['pt_incomes'], width, label='Part-Time', color='mediumseagreen')
    ax3.set_xticks(x);
    ax3.set_xticklabels(scales)
    ax3.set_title('Daily Income (Total)')
    ax3.legend()

    # 4. 司机效率 (Step Income) 对比
    ax4 = fig.add_subplot(2, 3, 5)
    ax4.bar(x - width / 2, res['ft_efficiency'], width, label='Full-Time', color='royalblue')
    ax4.bar(x + width / 2, res['pt_efficiency'], width, label='Part-Time', color='mediumseagreen')
    ax4.set_xticks(x);
    ax4.set_xticklabels(scales)
    ax4.set_title('Efficiency (Income per Active Step)')
    ax4.legend()

    plt.tight_layout()
    plt.savefig('experiment_final_report.png')
    print("Results saved to 'experiment_final_report.png'")
    plt.show()


if __name__ == '__main__':
    # 请确保路径正确
    sim_file = 'generator/simulator_driver_nums=400_hex_scaling=0.017031373249357308_weekday.pkl'

    if os.path.exists(sim_file):
        # 确保先运行过 shared_ppo.py 生成了模型文件
        run_comparison_experiment(sim_file)
    else:
        print(f"Error: Simulator file not found at {sim_file}")
