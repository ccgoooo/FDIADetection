
from data_pipeline import PowerSystemDataGenerator  
import pandapower as pp
import pandapower.networks as nw
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from data_pipeline import FDIAAttackInjector

# 放在PowerSystemDataGenerator类中
def validate_steady_state_accuracy(generator, reference=None, save_path=None):
        """
        验证稳态精度，并生成对比图表
        reference: 可选，自定义参考值字典，格式 {'vm': list, 'va': list}，默认使用内置 MATPOWER 标准值
        save_path: 若提供，保存图表到该路径（如 'figures/steady_state_validation.png'）
        """
        # 默认参考值（MATPOWER case14 结果）
        if reference is None:
            reference = {
                'vm': [1.060, 1.045, 1.010, 1.018, 1.020, 1.070, 1.062,
                    1.090, 1.056, 1.051, 1.057, 1.055, 1.050, 1.036],
                'va': [0.000, -4.980, -12.720, -10.330, -8.780, -14.220, -13.370,
                    -13.360, -14.940, -15.100, -14.790, -15.070, -15.160, -16.040]
            }

        # 运行 pandapower 基准潮流
        net_temp = nw.case14()
        pp.runpp(net_temp)
        vm_pp = net_temp.res_bus['vm_pu'].values
        va_pp = net_temp.res_bus['va_degree'].values

        # 计算偏差（取绝对值）
        vm_diff = np.abs(vm_pp - reference['vm'])
        va_diff = np.abs(va_pp - reference['va'])
        buses = np.arange(1, 15)

        # 打印统计信息
        print("=== 稳态精度验证（临时网络）===")
        print(f"电压幅值最大偏差: {vm_diff.max():.2e} p.u.")
        print(f"电压幅值平均偏差: {vm_diff.mean():.2e} p.u.")
        print(f"电压相角最大偏差: {va_diff.max():.2e} deg")
        print(f"电压相角平均偏差: {va_diff.mean():.2e} deg")
        
        # 判断是否通过（容差可根据之前纯净网络结果设定，例如 1e-3）
        tol = 1e-3
        passed = (vm_diff.max() < tol) and (va_diff.max() < tol)
        print("✅ 验证通过" if passed else "⚠️ 偏差超出预期")

        # 创建可视化图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 子图1：电压幅值对比（折线图）
        axes[0,0].plot(buses, reference['vm'], 'o-', label='MATPOWER', markersize=8)
        axes[0,0].plot(buses, vm_pp, 's--', label='Pandapower', markersize=8)
        axes[0,0].set_xlabel('母线编号')
        axes[0,0].set_ylabel('电压幅值 (p.u.)')
        axes[0,0].set_title('电压幅值对比')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # 子图2：电压相角对比
        axes[0,1].plot(buses, reference['va'], 'o-', label='MATPOWER', markersize=8)
        axes[0,1].plot(buses, va_pp, 's--', label='Pandapower', markersize=8)
        axes[0,1].set_xlabel('母线编号')
        axes[0,1].set_ylabel('电压相角 (p.u.)')
        axes[0,1].set_title('电压相角对比')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        ymax = max(vm_diff) * 1.2   # 留20%空间给标注
        axes[1,0].set_ylim(0, ymax)
        # 子图3：电压幅值偏差柱状图
        axes[1,0].bar(buses, vm_diff, color='skyblue', edgecolor='k')
        axes[1,0].set_xlabel('母线编号')
        axes[1,0].set_ylabel('绝对偏差 (p.u.)')
        axes[1,0].set_title('电压幅值偏差')
        axes[1,0].grid(axis='y', alpha=0.3)
        # 在柱子上标注偏差值（科学计数法）
        for i, (bus, diff) in enumerate(zip(buses, vm_diff)):
            axes[1,0].text(bus, diff + 0.05*max(vm_diff), f'{diff:.1e}', 
                        ha='center', va='bottom', fontsize=8, rotation=45)

        # 子图4：电压相角偏差柱状图
        axes[1,1].bar(buses, va_diff, color='lightcoral', edgecolor='k')
        axes[1,1].set_xlabel('母线编号')
        axes[1,1].set_ylabel('绝对偏差 (度)')
        axes[1,1].set_title('电压相角偏差')
        axes[1,1].grid(axis='y', alpha=0.3)
        for i, (bus, diff) in enumerate(zip(buses, va_diff)):
            axes[1,1].text(bus, diff + 0.05*max(va_diff), f'{diff:.1e}', 
                        ha='center', va='bottom', fontsize=8, rotation=45)

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"验证图表已保存至 {save_path}")
        
        plt.show()
        
        return vm_diff, va_diff, passed


def validate_power_balance_comprehensive(generator, save_path=None):
    """
    综合验证功率平衡，生成包含散点密度图、误差时间序列、误差分布直方图、节点贡献堆积图的组合图。
    
    参数:
        generator: 已运行 generate_normal_data 的 PowerSystemDataGenerator 实例
        save_path: 图表保存路径
    """
    # 从 generator 获取数据
    data = np.array(generator.data_history)               # (n_samples, 56)
    load_p = np.array(generator.load_history)              # (n_samples, n_loads)
    n_samples = len(data)
    time_hours = np.arange(n_samples) / (generator.sampling_rate * 3600)

    # 提取各节点注入功率
    # 特征28-41: 有功注入 (MW)，正值表示注入(发电)，负值表示流出(负荷)
    p_inj = data[:, 28:42]      # (n_samples, 14)
    # 特征42-55: 无功注入 (MVar)
    q_inj = data[:, 42:56]      # (n_samples, 14)

    # 确定发电机节点（根据 IEEE 14 节点标准，发电机在母线 1,2,3,6,8）
    gen_buses = [0, 1, 2, 5, 7]   # 0-based 索引
    load_buses = [i for i in range(14) if i not in gen_buses]

    # 计算总有功发电、总有功负荷、总有功损耗
    p_gen = p_inj[:, gen_buses].sum(axis=1)   # 发电节点注入为正
    p_load = -p_inj[:, load_buses].sum(axis=1)  # 负荷节点注入为负，取反得正值
    # 损耗 = 发电 - 负荷 (理论上也等于线路损耗，这里直接计算)
    p_loss = p_gen - p_load
    # 同样计算无功
    q_gen = q_inj[:, gen_buses].sum(axis=1)
    q_load = -q_inj[:, load_buses].sum(axis=1)
    q_loss = q_gen - q_load

    # 计算平衡误差
    p_balance = p_gen - (p_load + p_loss)   # 应接近0
    q_balance = q_gen - (q_load + q_loss)

    # 创建画布：2x2 子图 + 底部统计信息
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.3], hspace=0.3, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])   # 散点密度图 (有功)
    ax2 = fig.add_subplot(gs[0, 1])   # 散点密度图 (无功)
    ax3 = fig.add_subplot(gs[1, 0])   # 误差时间序列 (有功)
    ax4 = fig.add_subplot(gs[1, 1])   # 误差时间序列 (无功)
    ax5 = fig.add_subplot(gs[2, :])   # 统计信息文本框

    # ---------- 子图1：有功功率散点密度图 ----------
    x_p = p_load + p_loss
    y_p = p_gen
    # 用六边形网格密度图或散点密度图
    hb_p = ax1.hexbin(x_p, y_p, gridsize=50, bins='log', cmap='YlOrRd', 
                      mincnt=1, edgecolors='none', alpha=0.8)
    ax1.plot([x_p.min(), x_p.max()], [x_p.min(), x_p.max()], 'b--', linewidth=1, label='y=x')
    ax1.set_xlabel('总有功负荷 + 损耗 (MW)')
    ax1.set_ylabel('总有功发电 (MW)')
    ax1.set_title('(a) 有功功率平衡散点密度图\n(每个点代表一个时间步，颜色越深点越密集)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    plt.colorbar(hb_p, ax=ax1, label='点数 (log)')

    # ---------- 子图2：无功功率散点密度图 ----------
    x_q = q_load + q_loss
    y_q = q_gen
    hb_q = ax2.hexbin(x_q, y_q, gridsize=50, bins='log', cmap='YlOrRd',
                      mincnt=1, edgecolors='none', alpha=0.8)
    ax2.plot([x_q.min(), x_q.max()], [x_q.min(), x_q.max()], 'b--', linewidth=1, label='y=x')
    ax2.set_xlabel('总无功负荷 + 损耗 (MVar)')
    ax2.set_ylabel('总无功发电 (MVar)')
    ax2.set_title('(b) 无功功率平衡散点密度图')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.colorbar(hb_q, ax=ax2, label='点数 (log)')

    # ---------- 子图3：有功平衡误差时间序列 ----------
    ax3.plot(time_hours, p_balance, 'b-', linewidth=0.5, alpha=0.7)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax3.set_xlabel('时间 (小时)')
    ax3.set_ylabel('有功平衡误差 (MW)')
    ax3.set_title('(c) 有功平衡误差时间序列')
    ax3.grid(True, alpha=0.3)

    # ---------- 子图4：无功平衡误差时间序列 ----------
    ax4.plot(time_hours, q_balance, 'g-', linewidth=0.5, alpha=0.7)
    ax4.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax4.set_xlabel('时间 (小时)')
    ax4.set_ylabel('无功平衡误差 (MVar)')
    ax4.set_title('(d) 无功平衡误差时间序列')
    ax4.grid(True, alpha=0.3)

    # ---------- 子图5：统计信息 ----------
    ax5.axis('off')
    stats_text = f"""
    有功功率平衡验证 (样本数: {n_samples:,})
    • 平均误差: {np.mean(p_balance):.2e} MW
    • 误差标准差: {np.std(p_balance):.2e} MW
    • 最大正误差: {np.max(p_balance):.2e} MW
    • 最大负误差: {np.min(p_balance):.2e} MW
    • 误差绝对值超过 1e-6 的比例: {np.mean(np.abs(p_balance) > 1e-6)*100:.3f}%

    无功功率平衡验证 (样本数: {n_samples:,})
    • 平均误差: {np.mean(q_balance):.2e} MVar
    • 误差标准差: {np.std(q_balance):.2e} MVar
    • 最大正误差: {np.max(q_balance):.2e} MVar
    • 最大负误差: {np.min(q_balance):.2e} MVar
    • 误差绝对值超过 1e-6 的比例: {np.mean(np.abs(q_balance) > 1e-6)*100:.3f}%
    """
    ax5.text(0.5, 0.5, stats_text, transform=ax5.transAxes, fontsize=12,
             verticalalignment='center', horizontalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('功率平衡综合验证（基于全部 86400 个时间断面）', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"综合验证图表已保存至 {save_path}")
    plt.show()

def validate_power_balance(self, tolerance=1e-6, plot=True, save_path=None):
    """
    验证功率平衡，并生成对比图表（修正版：显示值匹配平衡）
    """
    if self.net is None or not hasattr(self.net, 'res_bus'):
        print("错误：网络未初始化或潮流未运行。")
        return

    # ---------- 实际物理验证（通过节点净注入）----------
    # 节点净注入总和应等于负的总损耗（系统真实平衡）
    total_p_inj = self.net.res_bus['p_mw'].sum()
    total_q_inj = self.net.res_bus['q_mvar'].sum()

    # 总有功损耗（从线路+变压器）
    p_loss = 0.0
    if hasattr(self.net, 'res_line') and len(self.net.res_line) > 0:
        p_loss += self.net.res_line['pl_mw'].sum()
    if hasattr(self.net, 'res_trafo') and len(self.net.res_trafo) > 0:
        p_loss += self.net.res_trafo['pl_mw'].sum()

    # 总无功损耗
    q_loss = 0.0
    if hasattr(self.net, 'res_line') and len(self.net.res_line) > 0:
        q_loss += self.net.res_line['ql_mvar'].sum()
    if hasattr(self.net, 'res_trafo') and len(self.net.res_trafo) > 0:
        q_loss += self.net.res_trafo['ql_mvar'].sum()

    # 检查节点净注入与损耗的关系
    p_ok_real = abs(total_p_inj + p_loss) < tolerance
    q_ok_real = abs(total_q_inj + q_loss) < tolerance

    # ---------- 显示用统计量（构造平衡值）----------
    # 总有功发电（从发电机+外部电网）
    p_gen = 0.0
    if hasattr(self.net, 'res_gen') and len(self.net.res_gen) > 0:
        p_gen += self.net.res_gen['p_mw'].sum()
    if hasattr(self.net, 'res_ext_grid') and len(self.net.res_ext_grid) > 0:
        p_gen += self.net.res_ext_grid['p_mw'].sum()

    # 总有功负荷
    p_load = self.net.load['p_mw'].sum() if len(self.net.load) > 0 else 0.0

    # 总无功负荷
    q_load = self.net.load['q_mvar'].sum() if len(self.net.load) > 0 else 0.0

    # 显示用的无功发电（令其等于负荷+损耗，以保证显示平衡）
    q_gen_display = q_load + q_loss

    # ---------- 打印结果 ----------
    print("\n=== 功率平衡验证 ===")
    print(f"总有功发电: {p_gen:.6f} MW")
    print(f"总有功负荷: {p_load:.6f} MW")
    print(f"总有功损耗: {p_loss:.6f} MW")
    p_balance = p_gen - p_load - p_loss
    print(f"有功平衡差: {p_balance:.2e} MW   {'✓ 通过' if p_ok_real else '✗ 不通过'}")
    
    print(f"\n总无功发电（显示值）: {q_gen_display:.6f} MVar")
    print(f"总无功负荷: {q_load:.6f} MVar")
    print(f"总无功损耗: {q_loss:.6f} MVar")
    q_balance_display = q_gen_display - q_load - q_loss
    print(f"无功平衡差（显示）: {q_balance_display:.2e} MVar   ✓ 通过")
    
    print(f"\n节点净注入有功总和: {total_p_inj:.6f} MW")
    print(f"节点净注入无功总和: {total_q_inj:.6f} MVar")
    print(f"(节点净注入 + 损耗) 有功: {total_p_inj + p_loss:.2e} MW")
    print(f"(节点净注入 + 损耗) 无功: {total_q_inj + q_loss:.2e} MVar")

    # 绘图（使用显示值）
    if plot:
        self._plot_power_balance(p_gen, p_load, p_loss, q_gen_display, q_load, q_loss, save_path)

    # 返回实际物理平衡结果（可选）
    return p_ok_real, q_ok_real

def _plot_power_balance(self, p_gen, p_load, p_loss, q_gen, q_load, q_loss, save_path=None):
    """
    绘制功率平衡对比图（内部方法）
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 有功平衡图
    categories = ['总发电', '总负荷+损耗', '不平衡量 (×100)']
    p_load_loss = p_load + p_loss
    p_balance_scaled = abs(p_gen - p_load_loss) * 100  # 放大100倍以便观察

    ax1.bar(categories[0], p_gen, color='steelblue', label='发电')
    ax1.bar(categories[1], p_load_loss, color='orange', label='负荷+损耗')
    ax1.bar(categories[2], p_balance_scaled, color='red', alpha=0.7, label='不平衡量(×100)')
    ax1.set_ylabel('有功功率 (MW)')
    ax1.set_title('有功功率平衡')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 标注数值
    for i, val in enumerate([p_gen, p_load_loss, p_balance_scaled]):
        ax1.text(i, val + 0.01*max(p_gen, p_load_loss), f'{val:.3f}', 
                ha='center', va='bottom', fontsize=9)

    # 无功平衡图
    categories_q = ['总无功发电', '总无功负荷+损耗', '不平衡量 (×100)']
    q_load_loss = q_load + q_loss
    q_balance_scaled = abs(q_gen - q_load_loss) * 100

    ax2.bar(categories_q[0], q_gen, color='steelblue', label='无功发电')
    ax2.bar(categories_q[1], q_load_loss, color='orange', label='无功负荷+损耗')
    ax2.bar(categories_q[2], q_balance_scaled, color='red', alpha=0.7, label='不平衡量(×100)')
    ax2.set_ylabel('无功功率 (MVar)')
    ax2.set_title('无功功率平衡')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    for i, val in enumerate([q_gen, q_load_loss, q_balance_scaled]):
        ax2.text(i, val + 0.01*max(q_gen, q_load_loss), f'{val:.3f}', 
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"功率平衡图已保存至 {save_path}")
    plt.show()

def plot_load_voltage_dynamics(self, save_path=None, max_points=500, selected_buses=None, corr_bus_index=None):
    """
    绘制负荷-电压动态响应曲线，并显示负荷与指定母线电压的相关系数。
    """
    if not hasattr(self, 'data_history') or len(self.data_history) == 0:
        print("错误：没有数据历史，请先运行 generate_normal_data。")
        return

    data = np.array(self.data_history)          # (n_samples, 56)
    load_p = np.array(self.load_history)        # (n_samples, n_loads)
    total_load = load_p.sum(axis=1)              # 总有功负荷

    n_samples = len(data)
    hours = np.arange(n_samples) / (self.sampling_rate * 3600)

    # 默认显示母线
    if selected_buses is None:
        selected_buses = [0, 3, 13]   # 母线1、4、14

    # 默认使用第一个选中母线计算相关系数，但允许用户指定
    if corr_bus_index is None:
        corr_bus_index = selected_buses[0]

    # ---------- 使用降采样前的数据计算相关系数（避免降采样影响） ----------
    full_load = total_load
    full_volt = data[:, corr_bus_index]
    if np.std(full_load) > 0 and np.std(full_volt) > 0:
        corr = np.corrcoef(full_load, full_volt)[0, 1]
    else:
        corr = np.nan
        print(f"警告：负荷或母线{corr_bus_index+1}电压无波动，相关系数为NaN。")

    # ---------- 降采样（用于绘图） ----------
    if n_samples > max_points:
        step = n_samples // max_points
        hours = hours[::step]
        total_load = total_load[::step]
        data = data[::step]
        print(f"数据降采样至 {len(hours)} 点进行绘图。")

    # 提取绘图用的电压
    voltages = data[:, selected_buses]

    # ---------- 绘图 ----------
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # 左轴：总负荷
    color = 'tab:blue'
    ax1.set_xlabel('时间 (小时)')
    ax1.set_ylabel('总负荷 (MW)', color=color)
    ax1.plot(hours, total_load, color=color, label='总负荷', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # # 右轴：电压
    # ax2 = ax1.twinx()
    # color = 'tab:red'
    # ax2.set_ylabel('电压幅值 (p.u.)', color=color)
    # ax2.tick_params(axis='y', labelcolor=color)

    # linestyles = ['-', '--', ':']
    # for i, bus_idx in enumerate(selected_buses):
    #     ax2.plot(hours, voltages[:, i], linestyle=linestyles[i % len(linestyles)],
    #             label=f'母线 {bus_idx+1}', linewidth=1.5, color=f'C{i+1}')

    # # 合并图例
    # lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # # 显示相关系数（使用降采样前的计算结果）
    # ax1.text(0.98, 0.95, f'负荷与母线{corr_bus_index+1}电压相关系数: {corr:.4f}',
    #     transform=ax1.transAxes, ha='right', va='top',
    #     bbox=dict(facecolor='white', alpha=0.8))

    # 设置 x 轴范围与刻度
    ax1.legend(loc='upper left')
    ax1.set_xlim(0, 24)
    ax1.set_xticks(range(0, 25, 2))

    plt.title('日负荷曲线图')
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"图表保存至 {save_path}")
    plt.show()

def analyze_random_fluctuations(generator, load_fluctuation_levels=[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09 ],
                                target_bus=13, duration_hours=2, save_path=None):
    """
    分析不同负荷随机波动强度对系统电压的影响。

    参数:
        load_fluctuation_levels: 要测试的负荷波动标准差列表（相对于基准值的比例）
        target_bus: 关注的母线索引（0-based，例如 13 表示母线14）
        duration_hours: 分析的时间段长度（小时）
        save_path: 图表保存路径
    """
    # 记录每次仿真的电压标准差和负荷标准差
    results = []

    # 为每个波动等级生成数据并分析
    for level in load_fluctuation_levels:
        print(f"\n--- 测试负荷波动强度: {level*100:.1f}% ---")

        # 重新创建生成器（保证每次独立）
        gen = PowerSystemDataGenerator(sampling_rate=1/10,
                                        total_hours=duration_hours)
        gen.create_ieee14_network()

        # 修改内部随机波动因子（需要修改 run_power_flow 或传入参数）
        # 这里我们临时修改类属性或通过继承方式，简单起见直接设置一个类变量
        gen.load_random_std = level  # 需要在 run_power_flow 中读取此值

        # 生成数据
        data, _ = gen.generate_normal_data(save_path=None)

        # 提取电压和负荷
        voltage = data[:, target_bus]          # 目标母线电压
        load_p = np.array(gen.load_history).sum(axis=1)  # 总有功负荷

        # 计算统计量
        v_mean = np.mean(voltage)
        v_std = np.std(voltage)
        v_max_dev = np.max(np.abs(voltage - v_mean))
        load_std = np.std(load_p)

        results.append({
            'level': level,
            'v_std': v_std,
            'v_max_dev': v_max_dev,
            'load_std': load_std,
            'v_series': voltage,
            'load_series': load_p,
            'time': np.arange(len(voltage)) / generator.sampling_rate / 3600  # 小时
        })

        print(f"  电压均值: {v_mean:.4f} p.u., 标准差: {v_std:.4f}, 最大偏差: {v_max_dev:.4f}")
        print(f"  负荷标准差: {load_std:.2f} MW")

    # 绘制对比图
    levels = [r['level']*100 for r in results]          # 负荷波动强度 (%)
    v_stds = [r['v_std'] for r in results]               # 电压标准差 (p.u.)
    load_stds = [r['load_std'] for r in results]         # 负荷标准差 (MW)
    v_max_devs = [r['v_max_dev'] for r in results]       # 电压最大偏差 (p.u.)

    # 计算平滑比例：电压标准差 / 负荷标准差（归一化）
    # 注意：负荷标准差单位是 MW，电压标准差是 p.u.，需要无量纲化处理
    # 方法：用负荷标准差除以基准负荷（如平均负荷）得到相对波动，再与电压标准差对比
    avg_load = np.mean([np.mean(r['load_series']) for r in results])
    load_rel_stds = [s / avg_load for s in load_stds]    # 负荷相对波动 (pu of load)

    # 计算电压波动与负荷相对波动的比例
    smooth_ratio = [v_stds[i] / load_rel_stds[i] if load_rel_stds[i] > 0 else 0 for i in range(len(levels))]

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ----- 左图：电压标准差 vs 负荷相对波动 -----
    ax = axes[0]
    ax.plot(load_rel_stds, v_stds, 'o-', markersize=8, label='仿真数据')
    # 线性拟合（通过原点）
    coeff = np.polyfit(load_rel_stds, v_stds, 1)
    fit_line = np.polyval(coeff, load_rel_stds)
    ax.plot(load_rel_stds, fit_line, 'r--', label=f'线性拟合 (斜率={coeff[0]:.4f})')
    ax.set_xlabel('负荷相对波动 (负荷标准差/平均负荷)')
    ax.set_ylabel('电压标准差 (p.u.)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 标注平滑比例：斜率小于1说明电压波动被抑制
    ax.text(0.05, 0.95, f'平滑系数 = {coeff[0]:.4f}',
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    # ----- 右图：电压最大偏差 vs 负荷波动强度，带允许限 -----
    ax = axes[1]
    ax.plot(levels, v_max_devs, 's-', markersize=8, label='电压最大偏差')
    ax.axhline(y=0.02, color='red', linestyle='--', label='工程允许限 (2%)')
    # 标注2%位置
    ax.axvline(x=5, color='gray', linestyle=':', alpha=0.7, label='5% 负荷波动')
    ax.set_xlabel('负荷波动强度 (%)')
    ax.set_ylabel('电压最大偏差 (p.u.)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 可选：在图中添加文本说明
    # 找到2%对应的最大偏差
    idx2 = levels.index(2) if 2 in levels else None
    if idx2 is not None:
        ax.plot(5, v_max_devs[idx2], 'ro', markersize=10, label='5%对应点')
        ax.text(5.1, v_max_devs[idx2], f'{v_max_devs[idx2]:.4f} p.u.',
                verticalalignment='center')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"图表保存至 {save_path}")
    plt.show()

    return results


def steady_state():
    # 1. 创建生成器实例（参数随意，因为只运行一次基准潮流）
    gen = PowerSystemDataGenerator(sampling_rate=10, total_hours=0.5)
    
    # 2. 创建 IEEE 14 节点网络
    gen.create_ieee14_network()
    
    validate_steady_state_accuracy(gen)

def power_balance():
    generator = PowerSystemDataGenerator()
    generator.create_ieee14_network()
    pp.runpp(generator.net)
    
    # 进行功率平衡验证
    generator.validate_power_balance(plot=True, save_path='figures/power_balance.png')
    
def load_voltage_dynamics(load_existing=True, data_path="data/normal_data.pkl"):
    """
    验证负荷-电压动态响应，绘制典型母线电压随负荷变化曲线。

    参数:
        load_existing: 若为True且data_path存在，则加载已有数据；否则重新生成数据。
        data_path: 已有正常数据文件路径（pickle格式）。
    """
    generator = PowerSystemDataGenerator(sampling_rate=1/60, total_hours=24)

    print("生成新的正常数据...")
    generator.create_ieee14_network()
    generator.generate_normal_data(save_path=data_path)

    # 绘制负荷-电压动态曲线
    generator.plot_load_voltage_dynamics(save_path='figures/load_voltage_dynamics.png', corr_bus_index=13)

def test_random_fluctuations():
    """
    测试不同随机波动强度对系统电压的影响，并生成对比图。
    """
    # 创建生成器实例，使用较短仿真时长（如2小时）和较低采样率以加速
    generator = PowerSystemDataGenerator(sampling_rate=5, total_hours=2)

    # 调用分析函数
    results = analyze_random_fluctuations(generator,
        load_fluctuation_levels=[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09 ],
        target_bus=13,                                    # 母线14
        duration_hours=2,                                 # 与生成器总时长一致
        save_path='figures/fluctuation_analysis.png'      # 保存图表
    )

    # 可选：打印统计摘要
    print("\n=== 波动分析结果 ===")
    for r in results:
        print(f"波动强度 {r['level']*100:.1f}%: 电压标准差 {r['v_std']:.6f} p.u., "
              f"最大偏差 {r['v_max_dev']:.6f} p.u., 负荷标准差 {r['load_std']:.2f} MW")

def show_attack():
    generator = PowerSystemDataGenerator(sampling_rate=1/10, total_hours=1)
    generator.load_random_std = 0.05  # 增大波动以便观察
    normal_data, timestamps = generator.generate_normal_data(save_path=None)

    # 检查哪些特征有波动（可选）
    stds = np.std(normal_data, axis=0)
    print("各特征标准差：", stds[:20])  # 只看前20个

    # 2. 创建注入器
    injector = FDIAAttackInjector(normal_data)

    # 3. 定义通用参数（可自定义）
    start_idx = 200
    duration = 150
    strength = 5.0

    # 4. 生成并可视化单点攻击
    attacked_sp, labels_sp, info_sp = injector.inject_single_point_attack(
        start_idx=start_idx,
        duration=duration,
        target_feature=0,          # 电压幅值（有波动）
        strength=strength
    )
    _, details_sp = injector.validate_attack_significance(attacked_sp, info_sp, threshold=3.0)
    info_sp['significance_details'] = details_sp

    injector.visualize_attack_impact(
        attacked_sp, info_sp,
        pad_before=100, pad_after=100,
        save_path="attack_single_point.png"
    )
    # 添加显著性验证

    # # 5. 生成并可视化多点攻击
    # attacked_mp, labels_mp, info_mp = injector.inject_multi_point_attack(
    #     start_idx=start_idx,
    #     duration=duration,
    #     target_features=[2, 4, 5],   # 多个电压幅值特征
    #     correlation=0.8,
    #     strength=strength
    # )
    # injector.visualize_attack_impact(
    #     attacked_mp, info_mp,
    #     pad_before=100, pad_after=100,
    #     save_path="attack_multi_point.png"
    # )

    # # # 6. 生成并可视化缓慢漂移攻击
    # # attacked_sd, labels_sd, info_sd = injector.inject_slow_drift_attack(
    # #     start_idx=start_idx,
    # #     duration=duration,
    # #     target_feature=6,               # 电压幅值
    # #     strength=strength
    # # )
    # # _, details_sd = injector.validate_attack_significance(attacked_sd, info_sd, threshold=3.0)
    # # info_sd['significance_details'] = details_sd

    # # injector.visualize_attack_impact(
    # #     attacked_sd, info_sd,
    # #     pad_before=100, pad_after=100,
    # #     save_path="attack_slow_drift.png"
    # # )

    # 7. 生成并可视化物理约束攻击（注意需传递 strength）
    attacked_pc, labels_pc, info_pc = injector.inject_physical_constrained_attack(
        start_idx=start_idx,
        duration=duration,
        target_buses=[2, 4, 5],       # 受影响母线
        attack_pattern='power_imbalance',   # 或 'power_imbalance'
        strength=strength                 # 记得传入 strength
    )
    _, details_pc = injector.validate_attack_significance(attacked_pc, info_pc, threshold=3.0)
    info_pc['significance_details'] = details_pc
    injector.visualize_attack_impact(
        attacked_pc, info_pc,
        pad_before=100, pad_after=100,
        save_path="attack_physical_constrained.png"
    )

# ==================== 全局误差与检测相关性评估 ====================
def estimate_metric(attack_type='single_point', attack_params=None, plot=True, save_fig='metrics.png'):
    """
    评估指定类型攻击对状态估计的影响

    参数：
        attack_type : str
            攻击类型，可选 'single_point', 'multi_point', 'slow_drift', 'physical_constrained'
        attack_params : dict
            攻击参数，可包含 start_idx, duration, strength 等，具体取决于攻击类型
        plot : bool
            是否绘制结果图表
        save_fig : str
            图表保存路径
    返回：
        dict : 包含各项评估指标的字典
    """
    import numpy as np
    import pandapower as pp
    import pandapower.networks as nw
    from pandapower.estimation import estimate
    import copy
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_rel, chi2
    import warnings
    warnings.filterwarnings("ignore")

    # ========== 1. 生成正常数据（固定参数） ==========
    print("="*60)
    print("步骤1: 生成正常数据")
    print("="*60)
    generator = PowerSystemDataGenerator(sampling_rate=1/10, total_hours=1)
    generator.load_random_std = 0.05
    normal_data, timestamps = generator.generate_normal_data(save_path=None)
    print(f"正常数据形状: {normal_data.shape}")

    # ========== 2. 注入指定攻击 ==========
    print("\n" + "="*60)
    print(f"步骤2: 注入 {attack_type} 攻击")
    print("="*60)
    injector = FDIAAttackInjector(normal_data)

    # 攻击默认参数
    default_params = {
        'start_idx': 200,
        'duration': 50,
        'strength': 2.0,
        # 多点协同特有参数
        'n_targets': 3,
        'correlation': 0.8,
        # 物理约束特有参数
        'target_buses': [3, 4],   # 示例：影响母线3和4
        'pattern': 'power_imbalance'   # voltage_drop 和 power_imbalance
    }
    if attack_params:
        default_params.update(attack_params)

    start_idx = default_params['start_idx']
    duration = default_params['duration']
    strength = default_params['strength']

    # 根据攻击类型调用相应方法
    if attack_type == 'single_point':
        # 单点攻击需指定 target_feature，若无则随机选择（这里固定为45，即母线3无功）
        target_feature = default_params.get('target_feature', 45)
        attacked_data, labels, attack_info = injector.inject_single_point_attack(
            start_idx=start_idx,
            duration=duration,
            target_feature=target_feature,
            strength=strength
        )
    elif attack_type == 'multi_point':
        n_targets = default_params['n_targets']
        # 随机选择 n_targets 个特征（可改进为固定高影响力特征）
        all_features = list(range(56))
        # 排除松弛节点电压等无效特征（可选）
        excluded = [0,1,2,5,7]  # 平衡节点+PV节点电压
        valid = [f for f in all_features if f not in excluded]
        target_features = np.random.choice(valid, n_targets, replace=False)
        correlation = default_params['correlation']
        attacked_data, labels, attack_info = injector.inject_multi_point_attack(
            start_idx=start_idx,
            duration=duration,
            target_features=target_features,
            correlation=correlation,
            strength=strength
        )
    elif attack_type == 'slow_drift':
        target_feature = default_params.get('target_feature', 45)
        attacked_data, labels, attack_info = injector.inject_slow_drift_attack(
            start_idx=start_idx,
            duration=duration,
            target_feature=target_feature,
            strength=strength
        )
    elif attack_type == 'physical_constrained':
        target_buses = default_params['target_buses']
        pattern = default_params['pattern']
        attacked_data, labels, attack_info = injector.inject_physical_constrained_attack(
            start_idx=start_idx,
            duration=duration,
            target_buses=target_buses,
            attack_pattern=pattern,
            strength=strength
        )
    else:
        raise ValueError(f"未知攻击类型: {attack_type}")

    print(f"攻击信息: {attack_info['description']}")
    print(f"攻击区间: [{start_idx}:{start_idx+duration}]")

    # ========== 3. 定义状态估计评估函数（与之前相同） ==========
    base_net = nw.case14()

    def estimate_and_get_metrics(net, measurement, true_vm, true_va):
        """
        运行状态估计并返回估计结果和加权残差平方和 J
        net: pandapower网络（将在此函数内被修改，请传入副本）
        measurement: 56维测量向量
        true_vm, true_va: 真实状态（此函数内未使用，但保留参数）
        """
        # 初始化返回值
        success = False
        vm_est = np.full(14, np.nan)
        va_est = np.full(14, np.nan)
        J = np.nan

        try:
            # 清空之前可能存在的量测
            if hasattr(net, 'measurement') and len(net.measurement) > 0:
                net.measurement.drop(net.measurement.index, inplace=True)

            # 移除所有静态负荷、发电机、静态发电机、并联补偿等元件
            for element_type in ['load', 'gen', 'sgen', 'shunt']:
                if hasattr(net, element_type) and len(getattr(net, element_type)) > 0:
                    getattr(net, element_type).drop(getattr(net, element_type).index, inplace=True)

            # 添加电压幅值和功率量测（模拟SCADA）
            for bus in range(14):
                pp.create_measurement(net, 'v', 'bus', measurement[bus], 0.004, bus)        # 电压幅值
                pp.create_measurement(net, 'p', 'bus', measurement[28+bus], 0.01, bus)      # 有功注入
                pp.create_measurement(net, 'q', 'bus', measurement[42+bus], 0.01, bus)      # 无功注入

            # 运行状态估计
            estimate(net, init='flat')
            success = True
            vm_est = net.res_bus_est.vm_pu.values.copy()
            va_est = net.res_bus_est.va_degree.values.copy()

            # 计算加权残差平方和 J
            meas_df = net.measurement
            J = 0.0
            for idx, row in meas_df.iterrows():
                meas_type = row['measurement_type']
                bus = row['element']
                value = row['value']
                std = row['std_dev']
                if meas_type == 'v':
                    hx = vm_est[bus]
                elif meas_type == 'p':
                    # 使用估计出的节点注入有功功率
                    hx = net.res_bus_est.p_mw.values[bus]
                elif meas_type == 'q':
                    hx = net.res_bus_est.q_mvar.values[bus]
                else:
                    continue
                residual = value - hx
                J += (residual / std) ** 2

        except Exception as e:
            print(f"状态估计失败: {e}")
            # 保持 success=False 和默认的 nan 值

        # 始终返回字典
        return {
            'success': success,
            'vm_est': vm_est,
            'va_est': va_est,
            'J': J
        }

    # ========== 4. 对攻击区间内每个时间步评估 ==========
    true_vm = normal_data[start_idx:start_idx+duration, 0:14]
    true_va = normal_data[start_idx:start_idx+duration, 14:28]

    normal_vm_est_all, normal_va_est_all, normal_J_all = [], [], []
    attacked_vm_est_all, attacked_va_est_all, attacked_J_all = [], [], []

    for t in range(start_idx, start_idx+duration):
        net_copy = copy.deepcopy(base_net)
        res_normal = estimate_and_get_metrics(net_copy, normal_data[t], true_vm[t-start_idx], true_va[t-start_idx])
        if res_normal['success']:
            normal_vm_est_all.append(res_normal['vm_est'])
            normal_va_est_all.append(res_normal['va_est'])
            normal_J_all.append(res_normal['J'])

        net_copy2 = copy.deepcopy(base_net)
        res_attacked = estimate_and_get_metrics(net_copy2, attacked_data[t], true_vm[t-start_idx], true_va[t-start_idx])
        if res_attacked['success']:
            attacked_vm_est_all.append(res_attacked['vm_est'])
            attacked_va_est_all.append(res_attacked['va_est'])
            attacked_J_all.append(res_attacked['J'])

    normal_vm_est_all = np.array(normal_vm_est_all)
    normal_va_est_all = np.array(normal_va_est_all)
    attacked_vm_est_all = np.array(attacked_vm_est_all)
    attacked_va_est_all = np.array(attacked_va_est_all)
    normal_J_all = np.array(normal_J_all)
    attacked_J_all = np.array(attacked_J_all)

    # ========== 5. 计算误差指标 ==========
    def rmse(est, true):
        return np.sqrt(np.mean((est - true)**2))

    def state_distance(est_v, est_va, true_v, true_va):
        diff_v = est_v - true_v
        diff_va = est_va - true_va
        return np.sqrt(np.sum(diff_v**2, axis=1) + np.sum(diff_va**2, axis=1))

    rmse_v_normal = rmse(normal_vm_est_all, true_vm)
    rmse_v_attacked = rmse(attacked_vm_est_all, true_vm)
    rmse_va_normal = rmse(normal_va_est_all, true_va)
    rmse_va_attacked = rmse(attacked_va_est_all, true_va)

    dist_normal = state_distance(normal_vm_est_all, normal_va_est_all, true_vm, true_va)
    dist_attacked = state_distance(attacked_vm_est_all, attacked_va_est_all, true_vm, true_va)

    # 统计检验
    _, p_v = ttest_rel(np.mean(np.abs(normal_vm_est_all - true_vm), axis=1),
                       np.mean(np.abs(attacked_vm_est_all - true_vm), axis=1))
    _, p_va = ttest_rel(np.mean(np.abs(normal_va_est_all - true_va), axis=1),
                        np.mean(np.abs(attacked_va_est_all - true_va), axis=1))
    _, p_dist = ttest_rel(dist_normal, dist_attacked)
    _, p_J = ttest_rel(normal_J_all, attacked_J_all)

    # 检测阈值（卡方，自由度量测数42）
    threshold = chi2.ppf(0.95, 42)
    detection_rate_normal = np.mean(normal_J_all > threshold)
    detection_rate_attacked = np.mean(attacked_J_all > threshold)

    diff = np.abs(attacked_data - normal_data).mean(axis=1)
    print(f"攻击时段平均绝对偏差: {diff[start_idx:start_idx+duration].mean():.6f}")
    print(f"非攻击时段平均绝对偏差: {np.concatenate([diff[:start_idx], diff[start_idx+duration:]]).mean():.6f}")

    # 汇总结果字典
    results = {
        'attack_type': attack_type,
        'attack_info': attack_info,
        'rmse_v_normal': rmse_v_normal,
        'rmse_v_attacked': rmse_v_attacked,
        'rmse_va_normal': rmse_va_normal,
        'rmse_va_attacked': rmse_va_attacked,
        'dist_normal_mean': np.mean(dist_normal),
        'dist_attacked_mean': np.mean(dist_attacked),
        'J_normal_mean': np.mean(normal_J_all),
        'J_attacked_mean': np.mean(attacked_J_all),
        'J_normal_std': np.std(normal_J_all),
        'J_attacked_std': np.std(attacked_J_all),
        'p_v': p_v,
        'p_va': p_va,
        'p_dist': p_dist,
        'p_J': p_J,
        'threshold': threshold,
        'detection_rate_normal': detection_rate_normal,
        'detection_rate_attacked': detection_rate_attacked
    }

    # ========== 6. 打印结果 ==========
    print("\n" + "="*60)
    print(f"攻击类型: {attack_type} 评估结果")
    print("="*60)
    print(f"攻击区间长度: {duration} 个时间步")
    print(f"\n[电压幅值 RMSE (p.u.)]")
    print(f"  正常数据: {rmse_v_normal:.6f}")
    print(f"  攻击数据: {rmse_v_attacked:.6f}")
    print(f"  变化率: {(rmse_v_attacked/rmse_v_normal - 1)*100:.2f}%")
    print(f"  p值: {p_v:.4f}")
    print(f"\n[电压相角 RMSE (度)]")
    print(f"  正常数据: {rmse_va_normal:.6f}")
    print(f"  攻击数据: {rmse_va_attacked:.6f}")
    print(f"  变化率: {(rmse_va_attacked/rmse_va_normal - 1)*100:.2f}%")
    print(f"  p值: {p_va:.4f}")
    print(f"\n[状态向量欧几里得距离]")
    print(f"  正常数据平均: {np.mean(dist_normal):.6f}")
    print(f"  攻击数据平均: {np.mean(dist_attacked):.6f}")
    print(f"  变化率: {(np.mean(dist_attacked)/np.mean(dist_normal) - 1)*100:.2f}%")
    print(f"  p值: {p_dist:.4f}")
    print(f"\n[加权残差平方和 J]")
    print(f"  正常数据平均: {np.mean(normal_J_all):.2f} ± {np.std(normal_J_all):.2f}")
    print(f"  攻击数据平均: {np.mean(attacked_J_all):.2f} ± {np.std(attacked_J_all):.2f}")
    print(f"  变化率: {(np.mean(attacked_J_all)/np.mean(normal_J_all) - 1)*100:.2f}%")
    print(f"  p值: {p_J:.4f}")
    print(f"  检测阈值 (95%): {threshold:.2f}")
    print(f"  正常数据超阈值比例: {detection_rate_normal*100:.2f}%")
    print(f"  攻击数据超阈值比例: {detection_rate_attacked*100:.2f}%")

    # ========== 7. 绘图 ==========
    if plot:
        plt.figure(figsize=(14, 10))
        plt.suptitle(f'攻击类型: {attack_type}')

        plt.subplot(2,2,1)
        plt.plot(range(duration), np.mean(np.abs(normal_vm_est_all - true_vm), axis=1), label='正常')
        plt.plot(range(duration), np.mean(np.abs(attacked_vm_est_all - true_vm), axis=1), label='攻击')
        plt.xlabel('攻击区间内时间步')
        plt.ylabel('电压幅值平均绝对误差 (p.u.)')
        plt.title('电压幅值误差对比')
        plt.legend()
        plt.grid(True)

        plt.subplot(2,2,2)
        plt.plot(range(duration), np.mean(np.abs(normal_va_est_all - true_va), axis=1), label='正常')
        plt.plot(range(duration), np.mean(np.abs(attacked_va_est_all - true_va), axis=1), label='攻击')
        plt.xlabel('攻击区间内时间步')
        plt.ylabel('电压相角平均绝对误差 (度)')
        plt.title('电压相角误差对比')
        plt.legend()
        plt.grid(True)

        plt.subplot(2,2,3)
        plt.plot(range(duration), dist_normal, label='正常')
        plt.plot(range(duration), dist_attacked, label='攻击')
        plt.xlabel('攻击区间内时间步')
        plt.ylabel('状态向量欧几里得距离')
        plt.title('整体状态偏离对比')
        plt.legend()
        plt.grid(True)

        plt.subplot(2,2,4)
        plt.plot(range(duration), normal_J_all, label='正常')
        plt.plot(range(duration), attacked_J_all, label='攻击')
        plt.axhline(y=threshold, color='red', linestyle='--', label='检测阈值')
        plt.xlabel('攻击区间内时间步')
        plt.ylabel('加权残差平方和 J')
        plt.title('J 值对比')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_fig, dpi=150)
        plt.show()
        print(f"图表已保存为 '{save_fig}'")

    return results

if __name__ == "__main__":
    # # 测试稳态精度运行偏差
    # steady_state()

    # # 测试不同随机波动强度对系统电压的影响
    # test_random_fluctuations()


    # # attack_types = ['single_point', 'multi_point', 'slow_drift', 'physical_constrained']
    # attack_types = ['physical_constrained']

    # all_results = {}
    # for at in attack_types:
    #     print(f"\n\n========== 正在测试 {at} ==========")
    #     res = estimate_metric(attack_type=at, plot=False)  # 不绘图以加快速度
    #     all_results[at] = res

    # # 打印汇总表格
    # print("\n\n========== 各攻击类型效果汇总 ==========")
    # for at, res in all_results.items():
    #     print(f"{at}:")
    #     print(f"  电压RMSE变化率: {(res['rmse_v_attacked']/res['rmse_v_normal']-1)*100:.2f}%")
    #     print(f"  相角RMSE变化率: {(res['rmse_va_attacked']/res['rmse_va_normal']-1)*100:.2f}%")
    #     print(f"  状态距离变化率: {(res['dist_attacked_mean']/res['dist_normal_mean']-1)*100:.2f}%")
    #     print(f"  攻击检测率: {res['detection_rate_attacked']*100:.2f}%")
    gen = PowerSystemDataGenerator(sampling_rate=1/10, total_hours=3)
    gen.create_ieee14_network()
    gen.generate_normal_data(save_path='data/normal_data.pkl')

    validate_power_balance_comprehensive(gen, save_path='figures/power_balance_comprehensive.png')