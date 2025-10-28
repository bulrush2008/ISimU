"""
Complete data processing pipeline example in English

VTK read -> Grid interpolation -> HDF5 storage
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_reader import VTKReader
from interpolator import GridInterpolator
from hdf5_storage import HDF5Storage
import numpy as np


def main():
    """Complete data processing pipeline"""
    print("=== ISimU Complete Data Processing Pipeline ===\n")

    # Configuration
    vtm_file = "../Data/vessel.000170.vtm"
    output_h5 = "../Data/output_vessel_170.h5"
    output_vtk = "../Data/output_vessel_170.vts"

    # Interpolation parameters
    grid_size = (32, 32, 32)
    interpolation_method = 'linear'

    print(f"Configuration:")
    print(f"  - Input file: {vtm_file}")
    print(f"  - Grid size: {grid_size}")
    print(f"  - Interpolation method: {interpolation_method}")
    print(f"  - Output HDF5: {output_h5}")
    print(f"  - Output VTK: {output_vtk}\n")

    try:
        # Step 1: Read VTK file
        print("Step 1: Reading VTK file...")
        reader = VTKReader()
        vtk_data = reader.read_vtm(vtm_file)

        # Get available field variables
        available_fields = reader.get_available_fields(vtk_data)
        print(f"  [OK] Successfully read VTK file")
        print(f"  - Number of data blocks: {vtk_data['num_blocks']}")
        print(f"  - Available field variables: {available_fields}")

        # Step 2: Grid interpolation
        print(f"\nStep 2: Performing grid interpolation...")
        interpolator = GridInterpolator(
            grid_size=grid_size,
            method=interpolation_method
        )

        # Select field variables to interpolate
        fields_to_interpolate = available_fields[:3] if len(available_fields) > 3 else available_fields
        print(f"  - Interpolating fields: {fields_to_interpolate}")

        interpolated_data = interpolator.interpolate(vtk_data, fields_to_interpolate)

        # Get interpolation statistics
        stats = interpolator.get_interpolation_statistics(interpolated_data)
        print(f"  [OK] Interpolation completed")
        print(f"  - Total grid points: {stats['total_points']:,}")

        for field_name, field_stats in stats['field_statistics'].items():
            print(f"    {field_name}: range[{field_stats['min']:.3e}, {field_stats['max']:.3e}], "
                  f"mean{field_stats['mean']:.3e}")

        # Step 3: Save as HDF5 format
        print(f"\nStep 3: Saving as HDF5 format...")
        storage = HDF5Storage()

        # Prepare metadata
        metadata = {
            'source_file': vtm_file,
            'grid_size': grid_size,
            'interpolation_method': interpolation_method,
            'original_fields': available_fields,
            'interpolated_fields': fields_to_interpolate,
            'description': 'ISimU interpolated data - vessel case time step 170'
        }

        storage.save(interpolated_data, output_h5, metadata)
        print(f"  [OK] HDF5 file saved")

        # Verify saved file
        file_info = storage.get_file_info(output_h5)
        print(f"  - File size: {file_info['total_data_size_mb']:.2f} MB")

        # Step 4: Convert to VTK format for visualization
        print(f"\nStep 4: Converting to VTK format...")
        try:
            storage.convert_to_vtk(output_h5, output_vtk)
            print(f"  [OK] VTK file saved")
            print(f"  - Can open with ParaView: {output_vtk}")
        except Exception as e:
            print(f"  VTK conversion failed: {e}")

        print(f"\n=== Processing Completed ===")
        print(f"Output files:")
        print(f"  - HDF5 data: {output_h5}")
        print(f"  - VTK visualization: {output_vtk}")

        return True

    except Exception as e:
        print(f"\n[ERROR] Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_validation():
    """Test data validation functionality"""
    print("\n=== Data Validation Test ===")

    try:
        from hdf5_storage import HDF5Storage

        # Load previously generated HDF5 file
        h5_file = "../Data/output_vessel_170.h5"
        if not os.path.exists(h5_file):
            print("  Skipping validation: HDF5 file not found")
            return

        storage = HDF5Storage()
        data = storage.load(h5_file)

        print(f"  [OK] Successfully loaded HDF5 file")
        print(f"  - Grid size: {data['grid_size']}")
        print(f"  - Bounds: {data['bounds']}")

        # Validate data integrity
        for field_name, field_data in data['fields'].items():
            nan_count = np.sum(np.isnan(field_data))
            inf_count = np.sum(np.isinf(field_data))
            zero_count = np.sum(field_data == 0)

            print(f"  - {field_name}:")
            print(f"    Shape: {field_data.shape}")
            print(f"    NaN points: {nan_count}")
            print(f"    Infinite points: {inf_count}")
            print(f"    Zero points: {zero_count}")

            if nan_count > 0 or inf_count > 0:
                print(f"    [WARNING] Data quality issues detected")
            else:
                print(f"    [OK] Data quality is good")

    except Exception as e:
        print(f"  [ERROR] Validation failed: {e}")


if __name__ == "__main__":
    success = main()

    if success:
        test_data_validation()

    if not success:
        sys.exit(1)