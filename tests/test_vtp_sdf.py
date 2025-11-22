"""
测试VTP几何的SDF计算
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sdf_utils import create_sdf_from_vmr_case
import numpy as np

def test_vtp_sdf():
    """测试VTP SDF计算功能"""
    print("=== VTP SDF Calculation Test ===\n")

    try:
        # 创建VTP SDF计算器
        print("1. Creating SDF from VMR geometry...")
        sdf = create_sdf_from_vmr_case("0007_H_AO_H")

        if sdf is None:
            print("  [ERROR] Failed to create SDF calculator")
            return False

        print(f"  [OK] SDF calculator created successfully")
        print(f"  - Vertices: {len(sdf.vertices):,}")
        print(f"  - Faces: {len(sdf.faces):,}")
        print(f"  - Geometry source: {getattr(sdf, 'geometry_source', 'Unknown')}")
        print()

        # 测试SDF计算
        print("2. Testing SDF calculation on sample points...")
        test_points = np.array([
            [0.0, 0.0, 0.0],    # 中心点
            [10.0, 10.0, 10.0], # 远离血管的点
            [-5.0, -3.0, -10.0], # 边界点
        ])

        sdf_values = sdf.compute_sdf(test_points, batch_size=10)

        print(f"  [OK] SDF calculation completed")
        for i, (point, sdf_val) in enumerate(zip(test_points, sdf_values)):
            location = "inside" if sdf_val > 0 else "outside"
            print(f"    Point {i}: {point} -> SDF={sdf_val:.3e} ({location})")
        print()

        # 测试掩码功能
        print("3. Testing inside/outside mask...")
        inside_mask, outside_mask = sdf.get_inside_outside_mask(test_points)

        print(f"  Inside points: {np.sum(inside_mask)}")
        print(f"  Outside points: {np.sum(outside_mask)}")
        print()

        print("=== VTP SDF Test Completed Successfully ===")
        return True

    except Exception as e:
        print(f"  [ERROR] VTP SDF test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vtp_sdf()
    sys.exit(0 if success else 1)