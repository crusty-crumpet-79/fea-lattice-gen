import sys
import os
import numpy as np
import pyvista as pv


# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from fieldlat import generate_adaptive_lattice

def test_watertight_lattice():
    # 1. Create a simple dummy mesh (Cube)
    # We use a simple bounds box
    mesh = pv.Cube(center=(0, 0, 0), x_length=1.0, y_length=1.0, z_length=1.0)
    # Triangulate and subdivide to have points for the field
    mesh = mesh.triangulate().subdivide(2)
    
    # 2. Add a scalar field
    field_name = 'stress'
    mesh.point_data[field_name] = np.ones(mesh.n_points) * 0.5 # Constant field
    
    # 3. Generate Lattice with padding (default)
    lattice = generate_adaptive_lattice(
        mesh,
        field_name=field_name,
        resolution=30, # Low resolution for speed
        base_scale=5.0,
        dense_scale=5.0,
        threshold=0.3,
        pad_width=2
    )
    
    # 4. Check for open edges
    # clean() might be needed to merge duplicate points from marching cubes if any, 
    # though marching cubes usually produces shared vertices if indexed correctly.
    # scikit-image marching_cubes returns unique vertices and faces, but we did:
    # surface_mesh = pv.PolyData(verts, pv_faces)
    # Let's clean it just in case to merge vertices
    lattice = lattice.clean()
    
    n_open = lattice.n_open_edges
    print(f"Number of open edges: {n_open}")
    
    if n_open > 0:
        # Save for inspection if failed
        lattice.save("failed_lattice.vtk")
        
    assert n_open == 0, f"Lattice is not watertight! Found {n_open} open edges."

if __name__ == "__main__":
    test_watertight_lattice()
    print("Test Passed: Lattice is watertight.")
