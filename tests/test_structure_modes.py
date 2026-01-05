import sys
import os
import numpy as np
import pyvista as pv
import pytest

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from fieldlat import generate_adaptive_lattice

def test_structure_modes():
    # 1. Create a simple dummy mesh
    mesh = pv.Cube(center=(0, 0, 0), x_length=1.0, y_length=1.0, z_length=1.0)
    mesh = mesh.triangulate().subdivide(1)
    
    # 2. Add a scalar field
    field_name = 'stress'
    mesh.point_data[field_name] = np.linspace(0, 1, mesh.n_points)
    
    modes = [
        ('sheet', 0.2), 
        ('strut', 0.0)
    ]
    
    for mode, thresh in modes:
        print(f"Testing structure_mode: {mode}")
        try:
            lattice = generate_adaptive_lattice(
                mesh,
                field_name=field_name,
                lattice_type='gyroid',
                structure_mode=mode,
                resolution=20, # Low resolution for speed
                base_scale=5.0,
                dense_scale=10.0,
                threshold=thresh,
                pad_width=1
            )
            
            assert lattice.n_points > 0, f"Lattice {mode} is empty!"
            assert lattice.n_cells > 0, f"Lattice {mode} has no cells!"
            
            # Check for watertightness
            # Note: Marching cubes usually produces closed meshes if padded correctly
            # But sometimes duplicate points occur, so we clean.
            # lattice = lattice.clean() # clean might remove degenerate faces but let's check open edges.
            
            # For strut mode with pad_width, it should be closed.
            # However, if the field is complex, sometimes marching cubes leaves open edges at the very boundary if not handled perfectly.
            # But our logic sets padding to 1.0 (Void).
            
            # n_open_edges check
            # For this check to be reliable, we might need `triangulate()` or `clean()`.
            lattice_clean = lattice.clean()
            # Depending on pyvista version/filters, `n_open_edges` property is available on PolyData.
            
            # Just asserting it runs and produces geometry is the main goal for this refactor verification.
            # But watertightness was a specific requirement ("Preserve Watertight Padding").
            
            # Let's check boundary.
            # Since we padded with 1.0 (Void), the mesh should be strictly inside the box (resolution bounds).
            # We can check if `lattice.n_open_edges == 0`.
            
            assert lattice_clean.n_open_edges == 0, f"Lattice {mode} is not watertight!"
            
            print(f"  -> {mode} passed.")
            
        except Exception as e:
            pytest.fail(f"Lattice generation failed for mode '{mode}': {e}")

if __name__ == "__main__":
    test_structure_modes()
