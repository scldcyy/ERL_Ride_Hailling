import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import numpy as np


# 生成模拟数据
def generate_simulated_results():
    # 1. 模拟 Pareto 前沿数据 1 (曲线 1)
    x_p1 = np.linspace(0.1, 10, 50)
    # 模拟一条负相关曲线，类似于原始图像中的 Pareto 前沿
    y_p1 = 1 / x_p1 + np.random.normal(0, 0.05, 50)

    # 2. 模拟另一个 Pareto 前沿数据 2 (曲线 2)
    x_p2 = np.linspace(0.1, 20, 70)
    # 模拟另一条具有不同比例和随机噪声的负相关曲线
    y_p2 = 25 - (x_p2 * 1.1) + np.random.normal(0, 1.2, 70)

    # 3. 模拟性能直方图数据 (直方图)
    # 模拟 5 个类别的性能指标，类似于原始图像中的柱状图
    categories = ['Quality', 'Efficiency', 'Interpretability', 'Convergence', 'Stability']
    values = np.random.randint(4, 9, len(categories))  # 随机生成一些性能值

    return x_p1, y_p1, x_p2, y_p2, categories, values


# 手动绘制神经网络架构图的函数
def draw_nn_architecture(ax):
    # 设置图表标题
    ax.set_title("Lower-layer MARL Behavior (NN)")
    # 隐藏坐标轴
    ax.axis('off')

    # 定义神经网络的层结构和标签
    # Input -> Hidden 1 -> Hidden 2 -> Output
    layer_sizes = [4, 6, 5, 2]  # 输入、隐藏 1、隐藏 2、输出层节点数
    layer_labels = ['Input', 'Hidden 1', 'Hidden 2', 'Output']

    # 节点和连接的绘图参数
    node_radius = 0.25
    h_spacing = 1.5
    v_spacing = 0.8

    # 计算所有节点的位置
    node_positions = []
    for i, size in enumerate(layer_sizes):
        x = i * h_spacing
        y_start = -(size - 1) * v_spacing / 2
        layer_nodes = []
        for j in range(size):
            y = y_start + j * v_spacing
            layer_nodes.append((x, y))
        node_positions.append(layer_nodes)

    # 绘制层与层之间的连接（所有节点对）
    for i in range(len(layer_sizes) - 1):
        curr_layer = node_positions[i]
        next_layer = node_positions[i + 1]
        for curr_node in curr_layer:
            for next_node in next_layer:
                line = plt.Line2D([curr_node[0], next_node[0]],
                                  [curr_node[1], next_node[1]],
                                  color='gray', alpha=0.3, lw=1)
                ax.add_line(line)

    # 绘制节点和标签
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(layer_sizes)))  # 使用调色板
    for i, (layer_nodes, size) in enumerate(zip(node_positions, layer_sizes)):
        color = colors[i]
        for j, node in enumerate(layer_nodes):
            # 绘制节点作为圆圈
            circle = patches.Circle(node, radius=node_radius, edgecolor='black', facecolor=color, alpha=0.7)
            ax.add_patch(circle)

            # 为某些节点添加 ID 标签（可选，类似于原始图像）
            if i == 0 and j == 0:
                ax.text(node[0], node[1], 's₁', ha='center', va='center', fontweight='bold', color='white')
            if i == len(layer_sizes) - 1 and j == 1:
                ax.text(node[0], node[1], 'a₁', ha='center', va='center', fontweight='bold', color='white')

        # 添加层标签
        ax.text(layer_nodes[0][0], layer_nodes[-1][1] - node_radius * 3, layer_labels[i], ha='center', va='top',
                fontweight='bold')

    # 添加 MARL 特有的反馈循环
    # 从一个输出节点到中间层（隐藏 2）
    out_node = node_positions[3][1]  # 使用第二个输出节点 'a₁'
    h2_node = node_positions[2][2]  # 使用中间的一个隐藏 2 节点

    # 使用弯曲的箭头绘制反馈循环
    loop_arrow = patches.FancyArrowPatch((out_node[0] + node_radius * 1.2, out_node[1]),
                                         (h2_node[0] - node_radius * 1.2, h2_node[1]),
                                         connectionstyle="arc3,rad=-0.7",  # 弯曲程度
                                         arrowstyle="->,head_length=5,head_width=3",
                                         color='purple', mutation_scale=20, lw=2, linestyle='dashed')
    ax.add_patch(loop_arrow)
    ax.text(h2_node[0] + (out_node[0] - h2_node[0]) / 2, out_node[1] - 0.3, "Policy\nFeedback", color='purple',
            ha='center')

    # 在输出节点附近绘制一个代表性的智能体图标（就像原始图像一样）
    agent_loc = (out_node[0] + 0.8, out_node[1])
    # 绘制一个简单的多边形作为智能体标记
    agent_marker = patches.RegularPolygon(agent_loc, 3, radius=0.4, edgecolor='black', facecolor='white',
                                          orientation=np.pi / 6, alpha=0.5)
    ax.add_patch(agent_marker)
    ax.text(agent_loc[0], agent_loc[1] - 0.5, "Agent₁\n(MARL)", ha='center')

    # 设置边界以确保所有内容可见
    ax.set_xlim(-0.5, h_spacing * (len(layer_sizes) - 1) + 1.5)
    ax.set_ylim(-3.5, 3.5)
    # 确保节点是圆的
    ax.set_aspect('equal')


def main():
    # 生成模拟数据
    x_p1, y_p1, x_p2, y_p2, categories, values = generate_simulated_results()

    # 创建 Gridspec 布局以垂直堆叠子图，并根据内容调整高度
    fig = plt.figure(figsize=(12, 16))
    # 4 行 1 列，神经网络图的高度比例较大
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 1.5])
    plt.subplots_adjust(hspace=0.4)  # 增加子图间距

    # 1. Pareto Front Analysis (曲线图 1)
    ax1 = fig.add_subplot(gs[0])
    # 绘制线图
    ax1.plot(x_p1, y_p1, 'o-', color='tab:blue', label='Front A')
    ax1.set_title("Theoretical Pareto Front Analysis")
    ax1.set_xlabel("Objective 1 (e.g., Cost)")
    ax1.set_ylabel("Objective 2 (e.g., Stability)")
    ax1.grid(True)
    ax1.legend()

    # 2. Experimental Verification (曲线图 2 - 散点图)
    ax2 = fig.add_subplot(gs[1])
    # 绘制散点图
    ax2.scatter(x_p2, y_p2, color='tab:orange', label='Front B')
    ax2.set_title("Experimental Pareto Front Verification")
    ax2.set_xlabel("Objective 1 (e.g., Efficiency)")
    ax2.set_ylabel("Objective 2 (e.g., Interpretability)")
    ax2.grid(True)
    ax2.legend()

    # 3. Performance Metrics (直方图)
    ax3 = fig.add_subplot(gs[2])
    # 绘制柱状图，并使用颜色映射
    rects = ax3.bar(categories, values, color=plt.cm.viridis(np.linspace(0.3, 0.7, len(categories))))
    ax3.set_title("Multi-Objective Performance Comparison")
    ax3.set_ylabel("Normalized Score")
    ax3.set_xlabel("Performance Metrics")
    # 设置 Y 轴范围以提供空间
    ax3.set_ylim(0, 10)

    # 为每个柱子添加值标签
    for rect in rects:
        height = rect.get_height()
        ax3.annotate('{}'.format(height),
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 点垂直偏移
                     textcoords="offset points",
                     ha='center', va='bottom')
    # 稍微旋转类别标签以避免重叠
    ax3.set_xticklabels(categories, rotation=15)

    # 4. Neural Network Architecture (神经网络图)
    ax4 = fig.add_subplot(gs[3])
    # 调用绘制神经网络的函数
    draw_nn_architecture(ax4)

    # 添加一个总标题（类似于原始图像）
    fig.suptitle("Simulated Methodology for Verification (Based on Input Image)", fontsize=16, fontweight='bold',
                 y=0.98)

    # 调整布局以适应总标题
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # 在交互式环境中运行（如果适用）
    # plt.show() 

    # 保存图像
    output_filename = "simulated_right_charts.png"
    plt.savefig(output_filename)
    print(f"图像已成功生成并保存为 '{output_filename}'。")


if __name__ == "__main__":
    main()