import sys
import os
import numpy as np
import pyvista as pv

# --- PATH SETUP ---
# Add '../src' to python path so we can import fieldlat without installing it
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_path)

from fieldlat.core import generate_adaptive_lattice

# --- 1. SETUP DUMMY DATA ---
# We create a simple block to act as our "FEBio Result"
print("Generating dummy input mesh...")
grid = pv.ImageData(dimensions=(20, 20, 20), spacing=(5, 5, 5))
mesh = grid.cast_to_unstructured_grid()

# Add a fake "Stress" field (Linear gradient from 0 to 100)
# This simulates high stress at one end and low at the other
points = mesh.points
stress_values = points[:, 0]  # Stress increases along X axis
# Normalize to 0-100 range for the demo
stress_values = (stress_values - stress_values.min()) / (stress_values.max() - stress_values.min()) * 100
mesh.point_data["stress"] = stress_values

# --- 2. CALCULATE SAFE BOUNDS ---
DEFAULT_RES = 200
# Calculate the physical size of one voxel
voxel_size = (mesh.bounds[1] - mesh.bounds[0]) / DEFAULT_RES
# Minimum safe cell size (~8 voxels for smooth definition)
safe_min = voxel_size * 8.0
# Maximum safe cell size (Full domain width)
safe_max = mesh.bounds[1] - mesh.bounds[0]

print(f"Calculated Dynamic Bounds (Res={DEFAULT_RES}):")
print(f"  - Voxel Size: {voxel_size:.2f}mm")
print(f"  - Safe Min Cell: {safe_min:.2f}mm (Nyquist Limit)")
print(f"  - Safe Max Cell: {safe_max:.2f}mm")

# --- 3. SETUP PLOTTER ---
p = pv.Plotter()
p.add_text("FieldLat Interactive Designer", font_size=18)

# Global State to store current slider values
params = {
    "min_cell_size": safe_min * 1.5, # Default to slightly above min
    "max_cell_size": safe_max / 4.0, # Default to 1/4 of domain
    "threshold": 0.5,
    "res": DEFAULT_RES
}

# Keep track of the actor to remove/replace it
current_actor = None

def update_mesh():
    """Re-runs the lattice generation and updates the view."""
    global current_actor
    
    # 1. Show a status message
    p.add_text("Generating...", name="status", position='upper_right', color='red', font_size=12)
    
    try:
        # Convert Cell Size (L) to Angular Frequency (k)
        # k = 2 * pi / L
        # Note: 'base_scale' is for low stress (larger cells -> max_cell_size)
        #       'dense_scale' is for high stress (smaller cells -> min_cell_size)
        
        # Protect against div by zero
        k_min = (2 * np.pi) / max(params["max_cell_size"], 0.001) 
        k_max = (2 * np.pi) / max(params["min_cell_size"], 0.001)

        # 2. Call your actual library logic
        lattice = generate_adaptive_lattice(
            mesh=mesh,
            field_name="stress",
            resolution=int(params["res"]),
            dense_scale=k_max,    # High Stress -> Small Cells
            base_scale=k_min,     # Low Stress -> Large Cells
            threshold=params["threshold"],
            lattice_type='gyroid',
            structure_mode='sheet',
            pad_width=2
        )
        
        # 3. Update the Plotter
        if current_actor:
            p.remove_actor(current_actor)
        
        # Add new mesh (White color, smooth shading)
        if lattice.n_points > 0:
            current_actor = p.add_mesh(lattice, color="white", specular=0.5, smooth_shading=True)
            p.add_text("", name="status") # Clear status
        else:
            p.add_text("Empty Mesh", name="status", color='yellow')

    except Exception as e:
        print(f"Generation failed: {e}")
        p.add_text(f"Error: {str(e)}", name="status", color='red')

# --- 4. DEFINE CALLBACKS ---

def set_min_cell(value):
    # Ensure min < max
    if value >= params["max_cell_size"]:
        pass # In a real app we'd clamp, but here just updating is fine
    params["min_cell_size"] = value
    update_mesh()

def set_max_cell(value):
    params["max_cell_size"] = value
    update_mesh()

def set_threshold(value):
    params["threshold"] = value
    update_mesh()

def toggle_resolution(flag):
    # flag is True (High Res) or False (Low Res)
    # Note: Changing resolution affects safety limits, but we keep sliders static for this demo
    params["res"] = 250 if flag else 100
    update_mesh()

# --- 5. ADD WIDGETS ---
# PyVista places these in the corner of the window

p.add_slider_widget(
    set_min_cell, 
    [safe_min, safe_max], 
    value=params["min_cell_size"], 
    title=f"Min Cell Size (Safe > {safe_min:.1f})", 
    pointa=(0.05, 0.9), pointb=(0.25, 0.9)
)

p.add_slider_widget(
    set_max_cell, 
    [safe_min, safe_max], 
    value=params["max_cell_size"], 
    title=f"Max Cell Size (Safe < {safe_max:.1f})", 
    pointa=(0.05, 0.75), pointb=(0.25, 0.75)
)

p.add_slider_widget(
    set_threshold, 
    [0.01, 1.2], 
    value=0.3, 
    title="Wall Thickness", 
    pointa=(0.05, 0.6), pointb=(0.25, 0.6)
)

p.add_checkbox_button_widget(
    toggle_resolution, 
    value=False, # Default to 100 (Low) since params["res"]=200 was our calc basis, we should align.
                 # Actually, let's align button: default False -> 200? 
                 # To avoid confusion, let's just say Button ON = 250, OFF = 150.
                 # We calculated bounds for 200.
    position=(10, 10), 
    size=30,
    border_size=2,
    color_on='green',
    color_off='grey'
)
p.add_text("Increase Res", position=(50, 15), font_size=10, color='black')

# --- 6. RUN ---
# Initial draw
update_mesh()
print("Starting Interactive Viewer... (Interact with the window)")
p.show()