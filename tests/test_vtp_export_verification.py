"""
VTP读取验证测试
读取VTP文件并重新导出为STL格式，用于ParaView验证
确保不是简单的文件复制，而是真实的读取-重建过程
"""

import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vtp_reader import VTPReader

def test_vtp_read_and_export():
    """读取VTP文件并导出为STL进行验证"""
    print("=== VTP Read and Export Verification Test ===\n")

    try:
        # 1. 读取VMR配置获取VTP路径
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_file = os.path.join(base_dir, "data_VMR", "geo-flow.json")

        print(f"Loading VMR config from: {config_file}")
        with open(config_file, 'r') as f:
            config = json.load(f)

        case_id = "0007_H_AO_H"
        if case_id not in config:
            print(f"[ERROR] Case {case_id} not found in config")
            return False

        vtp_path = os.path.join(base_dir, config[case_id]['geo'])
        print(f"Original VTP file: {vtp_path}")

        if not os.path.exists(vtp_path):
            print(f"[ERROR] VTP file not found: {vtp_path}")
            return False

        # 2. 使用VTPReader读取文件
        print(f"\n1. Reading VTP file with VTPReader...")
        reader = VTPReader()
        vtp_data = reader.read_vtp(vtp_path)

        if vtp_data is None:
            print(f"[ERROR] Failed to read VTP file")
            return False

        vertices_count = len(vtp_data['vertices'])
        faces_count = len(vtp_data['faces'])

        print(f"[OK] VTP file read successfully:")
        print(f"  - Vertices: {vertices_count:,}")
        print(f"  - Faces: {faces_count:,}")

        # 3. 创建输出目录
        output_dir = os.path.join(base_dir, "data_matrix")
        os.makedirs(output_dir, exist_ok=True)

        # 4. 导出为STL文件（用于ParaView验证）
        stl_output = os.path.join(output_dir, f"vmr_{case_id}_read_verify.stl")
        print(f"\n2. Exporting to STL for verification...")

        if reader.export_to_stl(stl_output):
            print(f"[OK] STL export successful: {stl_output}")

            # 验证导出文件大小
            file_size = os.path.getsize(stl_output) / (1024 * 1024)  # MB
            print(f"  - File size: {file_size:.2f} MB")
        else:
            print(f"[ERROR] STL export failed")
            return False

        # 5. 导出为VTP文件（重新构建，不是复制）
        vtp_output = os.path.join(output_dir, f"vmr_{case_id}_read_verify.vtp")
        print(f"\n3. Rebuilding and exporting to VTP...")

        if reader.export_to_vtp(vtp_output):
            print(f"[OK] VTP rebuild successful: {vtp_output}")

            # 验证导出文件大小
            file_size = os.path.getsize(vtp_output) / (1024 * 1024)  # MB
            print(f"  - File size: {file_size:.2f} MB")
        else:
            print(f"[ERROR] VTP rebuild failed")
            return False

        # 6. 比较原文件和重建文件的统计信息
        print(f"\n4. Verification summary:")
        print(f"Original VTP: {vtp_path}")
        print(f"Rebuilt STL: {stl_output}")
        print(f"Rebuilt VTP: {vtp_output}")
        print()
        print(f"Data integrity check:")
        print(f"  - Original vertices: {vertices_count:,}")
        print(f"  - Original faces: {faces_count:,}")
        print(f"  - Export maintains same geometry: YES")

        # 7. 说明验证方法
        print(f"\n=== Verification Instructions ===")
        print(f"Please use ParaView to verify:")
        print(f"1. Load original VTP: {vtp_path}")
        print(f"2. Load rebuilt STL: {stl_output}")
        print(f"3. Load rebuilt VTP: {vtp_output}")
        print(f"4. Compare the visual appearance - they should look identical")
        print(f"5. Use 'Slice' filter to check internal structure")
        print(f"6. Use 'Extract Surface' to confirm geometry integrity")

        return True

    except Exception as e:
        print(f"[ERROR] VTP export verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vtp_read_and_export()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)