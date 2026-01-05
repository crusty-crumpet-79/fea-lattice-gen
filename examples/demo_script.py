import numpy as np
import pyvista as pv
import os
import sys

# Ensure we can import the local package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from fieldlat import load_febio_vtk, generate_adaptive_lattice

def main():
    # 1. Define Input parameters
    input_file = 'input.vtk'
    field_name = 'stress'  # Replace with the actual scalar field name in your VTK
    
    print(f"--- FieldLat Demo ---")
    
    # Check if a dummy file needs to be created for the demo to run immediately
    if not os.path.exists(input_file):
        print(f"'{input_file}' not found. Creating a dummy VTK file for demonstration...")
        # Create a simple box mesh with some dummy data
        mesh = pv.Cube(center=(0, 0, 0), x_length=1.0, y_length=1.0, z_length=1.0)
        # Subdivide to give some resolution for the field (needs triangles)
        mesh = mesh.triangulate().subdivide(3) 
        # Add a dummy gradient field
        mesh.point_data[field_name] = mesh.points[:, 0] # Gradient along X
        mesh.save(input_file)
        print(f"Created '{input_file}' with field '{field_name}'.")

    # 2. Load Mesh
    try:
        print(f"Loading mesh from {input_file}...")
        mesh = load_febio_vtk(input_file, field_name)
        print(f"Mesh loaded. Points: {mesh.n_points}, Cells: {mesh.n_cells}")
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return

    # 3. Generate Lattice
    print("Generating adaptive lattice with Variable Cell Size...")
    try:
        lattice = generate_adaptive_lattice(
            mesh, 
            field_name=field_name,
            lattice_type='lidinoid', # Options: 'gyroid', 'diamond', 'primitive', 'lidinoid'
            resolution=60,         # Voxel grid resolution
            base_scale=10.0,       # Frequency for low-stress areas (larger cells)
            dense_scale=25.0,      # Frequency for high-stress areas (smaller cells)
            threshold=0.4          # Constant wall thickness
        )
        print("Lattice generation complete.")
        
        # 4. Visualization
        print("Visualizing result...")
        pl = pv.Plotter()
        pl.add_mesh(mesh, style='wireframe', color='black', opacity=0.1, label='Input Domain')
        pl.add_mesh(lattice, color='white', smooth_shading=True, label='Gyroid Lattice (Sheet)')
        pl.add_legend()
        pl.show()
        
        # Optional: Save result
        output_file = 'output_lattice.stl'
        lattice.save(output_file)
        print(f"Saved lattice to {output_file}")

        # 5. Generate Strut Lattice (Example)
        print("\nGenerating Strut-based lattice...")
        lattice_strut = generate_adaptive_lattice(
            mesh,
            field_name=field_name,
            lattice_type='gyroid',
            structure_mode='strut',  # New mode
            resolution=60,
            base_scale=8.0,
            dense_scale=15.0,
            threshold=0.0          # 0.0 implies 50/50 volume split for TPMS
        )
        print("Strut lattice generation complete.")
        
        pl2 = pv.Plotter()
        pl2.add_mesh(mesh, style='wireframe', color='black', opacity=0.1, label='Input Domain')
        pl2.add_mesh(lattice_strut, color='orange', smooth_shading=True, label='Gyroid Lattice (Strut)')
        pl2.add_legend()
        pl2.show()

    except Exception as e:
        print(f"Error generating lattice: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
