# test_convergence.py
import pandapower as pp
import pandapower.networks as nw
import numpy as np

def test_ieee14_convergence():
    """测试IEEE14系统的收敛性"""
    net = nw.case14()
    
    print("测试基准状态收敛性...")
    try:
        pp.runpp(net)
        print("✓ 基准状态收敛")
    except:
        print("✗ 基准状态不收敛")
        return False
    
    # 测试负载变化
    test_cases = [
        ("轻载", 0.5),
        ("正常", 1.0),
        ("重载", 1.5),
        ("过载", 2.0),
    ]
    
    for case_name, factor in test_cases:
        test_net = nw.case14()
        
        # 调整所有负载
        for load_idx in test_net.load.index:
            original_p = test_net.load.at[load_idx, 'p_mw']
            test_net.load.at[load_idx, 'p_mw'] = original_p * factor
            
            if 'q_mvar' in test_net.load.columns:
                original_q = test_net.load.at[load_idx, 'q_mvar']
                test_net.load.at[load_idx, 'q_mvar'] = original_q * factor
        
        try:
            pp.runpp(test_net, max_iteration=100)
            print(f"✓ {case_name} (x{factor}) 收敛")
        except Exception as e:
            print(f"✗ {case_name} (x{factor}) 不收敛: {e}")
    
    return True

if __name__ == "__main__":
    test_ieee14_convergence()