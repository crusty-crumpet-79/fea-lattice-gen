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

# --- 2. SETUP PLOTTER ---
p = pv.Plotter()
p.add_text("FieldLat Interactive Designer", font_size=18)

# Global State to store current slider values
params = {
    "dense_scale": 1.2,
    "base_scale": 0.2,
    "threshold": 0.5,
    "res": 200  # Start with moderate resolution
}

# Keep track of the actor to remove/replace it
current_actor = None

def update_mesh():
    """Re-runs the lattice generation and updates the view."""
    global current_actor
    
    # 1. Show a status message
    p.add_text("Generating...", name="status", position='upper_right', color='red', font_size=12)
    
    try:
        # 2. Call your actual library logic
        # Note: We use the current params dictionary
        lattice = generate_adaptive_lattice(
            mesh=mesh,
            field_name="stress",
            resolution=int(params["res"]),  # Use slider resolution
            dense_scale=params["dense_scale"],
            base_scale=params["base_scale"],
            threshold=params["threshold"],
            lattice_type='gyroid',   # You could add a dropdown for this later!
            structure_mode='sheet',
            pad_width=2              # Force watertight boundary
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

# --- 3. DEFINE CALLBACKS ---
# These functions are called when you move the sliders

def set_dense_scale(value):
    params["dense_scale"] = value
    update_mesh()

def set_base_scale(value):
    params["base_scale"] = value
    update_mesh()

def set_threshold(value):
    params["threshold"] = value
    update_mesh()

def toggle_resolution(flag):
    # flag is True (High Res) or False (Low Res)
    params["res"] = 80 if flag else 30
    update_mesh()

# --- 4. ADD WIDGETS ---
# PyVista places these in the corner of the window

p.add_slider_widget(
    set_dense_scale, 
    [0.1, 5.0], 
    value=1.2, 
    title="High Stress Freq (k_max)", 
    pointa=(0.05, 0.9), pointb=(0.25, 0.9) # Screen coordinates
)

p.add_slider_widget(
    set_base_scale, 
    [0.1, 5.0], 
    value=0.2, 
    title="Low Stress Freq (k_min)", 
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
    value=False, 
    position=(10, 10), 
    size=30,
    border_size=2,
    color_on='green',
    color_off='grey'
)
p.add_text("High Res Mode", position=(50, 15), font_size=10, color='black')

# --- 5. RUN ---
# Initial draw
update_mesh()
print("Starting Interactive Viewer... (Interact with the window)")
p.show()
