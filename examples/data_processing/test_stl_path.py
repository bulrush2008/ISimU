"""
Simple test to verify STL path resolution works
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from stl_reader import load_portal_vein_geometry

def test_stl_path_resolution():
    """Test that STL path resolution works correctly"""
    print("=== STL Path Resolution Test ===\n")

    print("Testing STL file loading with automatic path detection...")
    stl_data = load_portal_vein_geometry()

    if stl_data is None:
        print("✗ Failed to load STL file")
        return False
    else:
        print("✓ Successfully loaded STL file")
        print(f"  - Vertices: {stl_data['num_vertices']:,}")
        print(f"  - Faces: {stl_data['num_faces']:,}")
        print(f"  - Scale factor: {stl_data['scale_factor']}")
        print(f"  - Watertight: {stl_data['is_watertight']}")
        print(f"  - Scaled bounds: {stl_data['scaled_bounds']}")
        return True

if __name__ == "__main__":
    success = test_stl_path_resolution()
    if not success:
        sys.exit(1)