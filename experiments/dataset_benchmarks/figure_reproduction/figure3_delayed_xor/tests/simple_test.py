#!/usr/bin/env python3
"""
简化的测试脚本 - 只测试核心功能
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def test_basic_components():
    """测试基本组件"""
    print("Testing basic components...")
    
    # 测试DelayedXORDataset
    try:
        from reproduce_figure3 import DelayedXORDataset
        dataset = DelayedXORDataset(10, 20, 5, torch.device('cpu'))
        print(f"✓ DelayedXORDataset: {dataset.data.shape}")
    except Exception as e:
        print(f"❌ DelayedXORDataset error: {e}")
        return False
    
    # 测试VanillaSNN
    try:
        from reproduce_figure3 import VanillaSNN
        model = VanillaSNN(2, 16, 2)
        x = torch.randn(5, 2)
        output = model(x)
        print(f"✓ VanillaSNN: {output.shape}")
    except Exception as e:
        print(f"❌ VanillaSNN error: {e}")
        return False
    
    return True

def test_plotting():
    """测试绘图功能"""
    print("Testing plotting...")
    
    try:
        from reproduce_figure3 import plot_figure3_panel_a
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        plot_figure3_panel_a(ax)
        plt.savefig('test_simple.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Panel A plotting successful")
        return True
    except Exception as e:
        print(f"❌ Plotting error: {e}")
        return False

def test_demo_mode():
    """测试演示模式"""
    print("Testing demo mode...")
    
    try:
        from reproduce_figure3 import reproduce_figure3
        results = reproduce_figure3(
            save_path='simple_demo.png',
            run_experiments=False,
            use_existing_data=True
        )
        print("✓ Demo mode successful")
        print(f"✓ Keys: {list(results.keys())}")
        return True
    except Exception as e:
        print(f"❌ Demo error: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("Simple DH-SNN Figure 3 Test")
    print("=" * 50)
    
    tests = [
        ("Basic Components", test_basic_components),
        ("Plotting", test_plotting),
        ("Demo Mode", test_demo_mode)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n{name}:")
        try:
            if test_func():
                passed += 1
                print(f"✓ {name} PASSED")
            else:
                print(f"❌ {name} FAILED")
        except Exception as e:
            print(f"❌ {name} FAILED with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
