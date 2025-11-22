"""
测试VTP阅读器是否正确读入VTP文件
验证顶点、面片数据的完整性和正确性
"""

import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vtp_reader import VTPReader
import json

def test_vtp_reader():
    """测试VTP阅读器的功能"""
    print("=== VTP Reader Test ===\n")

    try:
        # 1. 读取VMR配置文件获取VTP路径
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_file = os.path.join(base_dir, "data_VMR", "geo-flow.json")

        print(f"Loading VMR config from: {config_file}")
        with open(config_file, 'r') as f:
            config = json.load(f)

        case_id = "0007_H_AO_H"
        if case_id not in config:
            print(f"[ERROR] Case {case_id} not found in config")
            return False

        geometry_path = os.path.join(base_dir, config[case_id]['geo'])
        print(f"VTP file path: {geometry_path}")

        if not os.path.exists(geometry_path):
            print(f"[ERROR] VTP file not found: {geometry_path}")
            return False

        # 2. 创建VTP阅读器并读取文件
        print("\n1. Reading VTP file...")
        reader = VTPReader()
        vtp_data = reader.read_vtp(geometry_path)

        if vtp_data is None:
            print("[ERROR] Failed to read VTP file")
            return False

        print(f"[OK] VTP file read successfully")

        # 3. 检查基本数据结构
        print("\n2. Checking data structure...")
        required_keys = ['vertices', 'faces', 'num_vertices', 'num_faces', 'scaled_bounds']
        for key in required_keys:
            if key not in vtp_data:
                print(f"[ERROR] Missing key: {key}")
                return False

        print(f"[OK] All required keys present")

        # 4. 详细分析顶点数据
        print("\n3. Analyzing vertex data...")
        vertices = vtp_data['vertices']
        num_vertices = vtp_data['num_vertices']

        print(f"  - Vertex count: {num_vertices:,}")
        print(f"  - Vertex array shape: {vertices.shape}")
        print(f"  - Expected shape: ({num_vertices}, 3)")

        if vertices.shape != (num_vertices, 3):
            print(f"[ERROR] Vertex shape mismatch")
            return False

        # 检查顶点数据的有效性
        print(f"  - Vertex coordinate ranges:")
        for i, axis in enumerate(['X', 'Y', 'Z']):
            coord_min = np.min(vertices[:, i])
            coord_max = np.max(vertices[:, i])
            coord_mean = np.mean(vertices[:, i])
            print(f"    {axis}: [{coord_min:.6f}, {coord_max:.6f}], mean: {coord_mean:.6f}")

        # 检查是否有NaN或无穷大值
        nan_count = np.sum(np.isnan(vertices))
        inf_count = np.sum(np.isinf(vertices))
        print(f"  - NaN values: {nan_count}")
        print(f"  - Infinite values: {inf_count}")

        if nan_count > 0 or inf_count > 0:
            print("[WARNING] Invalid vertex coordinates found")

        # 5. 详细分析面片数据
        print("\n4. Analyzing face data...")
        faces = vtp_data['faces']
        num_faces = vtp_data['num_faces']

        print(f"  - Face count: {num_faces:,}")
        print(f"  - Face array shape: {faces.shape}")
        print(f"  - Expected shape: ({num_faces}, 3)")

        if faces.shape != (num_faces, 3):
            print(f"[ERROR] Face shape mismatch")
            return False

        # 检查面片索引的有效性
        print(f"  - Face index ranges:")
        for i, corner in enumerate(['V0', 'V1', 'V2']):
            idx_min = np.min(faces[:, i])
            idx_max = np.max(faces[:, i])
            idx_mean = np.mean(faces[:, i])
            print(f"    {corner}: [{idx_min:.0f}, {idx_max:.0f}], mean: {idx_mean:.1f}")

        # 检查索引是否在有效范围内
        invalid_indices = np.where((faces < 0) | (faces >= num_vertices))[0]
        print(f"  - Invalid face indices: {len(invalid_indices)}")

        if len(invalid_indices) > 0:
            print(f"[ERROR] Found {len(invalid_indices)} invalid face indices")
            return False

        # 6. 分析边界框
        print("\n5. Analyzing bounds...")
        scaled_bounds = vtp_data['scaled_bounds']
        print(f"  - Bounds from VTP: [{scaled_bounds[0]}, {scaled_bounds[1]}]")
        print(f"  - Calculated bounds: [{np.min(vertices, axis=0)}, {np.max(vertices, axis=0)}]")

        # 7. 几何质量检查
        print("\n6. Geometry quality checks...")

        # 检查重复顶点
        unique_vertices = np.unique(vertices, axis=0)
        duplicate_count = num_vertices - len(unique_vertices)
        print(f"  - Duplicate vertices: {duplicate_count:,}")

        # 检查退化三角形（面积为0的三角形）
        face_vertices = vertices[faces]
        edge1 = face_vertices[:, 1] - face_vertices[:, 0]
        edge2 = face_vertices[:, 2] - face_vertices[:, 0]
        cross_products = np.cross(edge1, edge2)
        triangle_areas = 0.5 * np.linalg.norm(cross_products, axis=1)

        zero_area_count = np.sum(triangle_areas < 1e-12)
        print(f"  - Degenerate triangles (zero area): {zero_area_count:,}")
        print(f"  - Triangle area statistics:")
        print(f"    - Min area: {np.min(triangle_areas):.6e}")
        print(f"    - Max area: {np.max(triangle_areas):.6e}")
        print(f"    - Mean area: {np.mean(triangle_areas):.6e}")

        # 8. 数据一致性检查
        print("\n7. Data consistency checks...")

        # 检查面的法向量一致性
        face_normals = cross_products / (2.0 * triangle_areas.reshape(-1, 1) + 1e-12)
        normal_lengths = np.linalg.norm(face_normals, axis=1)
        invalid_normals = np.sum(np.isnan(normal_lengths) | np.isinf(normal_lengths))
        print(f"  - Invalid face normals: {invalid_normals}")

        # 9. 总结
        print("\n=== VTP Reader Test Summary ===")
        print(f"✅ Successfully read VTP file: {os.path.basename(geometry_path)}")
        print(f"✅ Vertices: {num_vertices:,}")
        print(f"✅ Faces: {num_faces:,}")
        print(f"✅ Duplicate vertices: {duplicate_count:,}")
        print(f"✅ Degenerate triangles: {zero_area_count:,}")
        print(f"✅ Invalid coordinates: {nan_count + inf_count}")
        print(f"✅ Invalid face indices: {len(invalid_indices)}")

        # 评估数据质量
        quality_issues = 0
        if duplicate_count > num_vertices * 0.01:  # 超过1%的重复顶点
            quality_issues += 1
        if zero_area_count > num_faces * 0.01:  # 超过1%的退化三角形
            quality_issues += 1
        if nan_count + inf_count > 0:
            quality_issues += 1
        if len(invalid_indices) > 0:
            quality_issues += 1

        if quality_issues == 0:
            print("🎉 VTP data quality: EXCELLENT")
            return True
        elif quality_issues <= 2:
            print("⚠️ VTP data quality: ACCEPTABLE (some minor issues)")
            return True
        else:
            print("❌ VTP data quality: POOR (significant issues found)")
            return False

    except Exception as e:
        print(f"[ERROR] VTP reader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vtp_reader()
    sys.exit(0 if success else 1)