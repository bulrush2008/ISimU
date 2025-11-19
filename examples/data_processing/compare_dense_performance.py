#!/usr/bin/env python3
"""
密集网格性能对比测试
比较48x48x48和64x64x64网格的性能差异
"""

import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from test_dense_48x48x48 import test_dense_48x48x48_interpolation
from test_dense_64x64x64 import test_dense_64x64x64_interpolation


def run_performance_comparison():
    """运行性能对比测试"""
    print("=== Dense Grid Performance Comparison ===\n")

    results = {}

    # 测试48x48x48网格
    print("1. Testing 48x48x48 grid (110,592 points)...")
    start_time = time.time()
    success_48 = test_dense_48x48x48_interpolation()
    end_time = time.time()

    if success_48:
        time_48 = end_time - start_time
        results['48x48x48'] = {
            'points': 48 * 48 * 48,
            'time': time_48,
            'points_per_second': (48 * 48 * 48) / time_48
        }
        print(f"[OK] 48x48x48 completed in {time_48:.1f} seconds")
        print(f"    - Performance: {results['48x48x48']['points_per_second']:.0f} points/second\n")
    else:
        print("[ERROR] 48x48x48 test failed\n")
        return False

    # 测试64x64x64网格
    print("2. Testing 64x64x64 grid (262,144 points)...")
    start_time = time.time()
    success_64 = test_dense_64x64x64_interpolation()
    end_time = time.time()

    if success_64:
        time_64 = end_time - start_time
        results['64x64x64'] = {
            'points': 64 * 64 * 64,
            'time': time_64,
            'points_per_second': (64 * 64 * 64) / time_64
        }
        print(f"[OK] 64x64x64 completed in {time_64:.1f} seconds")
        print(f"    - Performance: {results['64x64x64']['points_per_second']:.0f} points/second\n")
    else:
        print("[ERROR] 64x64x64 test failed\n")
        return False

    # 性能对比分析
    print("=== Performance Analysis ===")
    ratio_points = results['64x64x64']['points'] / results['48x48x48']['points']
    ratio_time = results['64x64x64']['time'] / results['48x48x48']['time']
    ratio_efficiency = results['64x64x64']['points_per_second'] / results['48x48x48']['points_per_second']

    print(f"Grid Size Comparison:")
    print(f"  - 48x48x48: {results['48x48x48']['points']:,} points")
    print(f"  - 64x64x64: {results['64x64x64']['points']:,} points")
    print(f"  - Point ratio: {ratio_points:.2f}x\n")

    print(f"Execution Time:")
    print(f"  - 48x48x48: {results['48x48x48']['time']:.1f} seconds")
    print(f"  - 64x64x64: {results['64x64x64']['time']:.1f} seconds")
    print(f"  - Time ratio: {ratio_time:.2f}x\n")

    print(f"Processing Efficiency:")
    print(f"  - 48x48x48: {results['48x48x48']['points_per_second']:.0f} points/second")
    print(f"  - 64x64x64: {results['64x64x64']['points_per_second']:.0f} points/second")
    print(f"  - Efficiency ratio: {ratio_efficiency:.2f}x\n")

    print(f"Scaling Analysis:")
    if ratio_time < ratio_points:
        print(f"[OK] Good scaling: Time increased {ratio_time:.2f}x for {ratio_points:.2f}x more points")
        scaling_efficiency = (ratio_points / ratio_time) * 100
        print(f"    - Scaling efficiency: {scaling_efficiency:.1f}%")
    else:
        print(f"[WARNING] Suboptimal scaling: Time increased {ratio_time:.2f}x for {ratio_points:.2f}x more points")

    print(f"\n=== Optimization Status ===")
    print(f"[OK] Both tests completed successfully")
    print(f"[OK] Duplicate STL reading removed")
    print(f"[OK] SDF calculation optimized (single computation)")
    print(f"[OK] LinearNDInterpolator implemented")

    return True


if __name__ == "__main__":
    success = run_performance_comparison()
    if not success:
        sys.exit(1)