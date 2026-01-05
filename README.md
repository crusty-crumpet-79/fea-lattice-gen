# FieldLat: Field-Driven Adaptive Lattice Generation

FieldLat is a Python library for generating functionally graded lattice structures based on scalar fields (e.g., stress, strain, or temperature data from FEA simulations).

Unlike simple thickness modulation, FieldLat implements **Variable Cell Size** via lattice blending. This allows for smooth transitions between low-density (large cell) and high-density (small cell) lattice regions while maintaining a constant wall thickness, optimizing structures for both weight and mechanical performance.

## Key Features

- **Multiple Topologies**: Supports **Gyroid**, **Diamond** (Schwarz D), **Primitive** (Schwarz P), and **Lidinoid** TPMS structures.
- **Sheet & Strut Modes**: Choose between double-walled (Sheet) or solid skeletal (Strut) network topologies.
- **Variable Cell Size**: Dynamically adjusts pore size based on field intensity.
- **Lattice Blending**: Smoothly interpolates between different lattice frequencies.
- **Field Mapping**: Automatically normalizes input scalar fields (stress/displacement) to control lattice density.
- **Watertight Results**: Includes padding options to ensure the final mesh is closed and printable.
- **PyVista Integration**: Built on PyVista for efficient 3D processing and visualization.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/FieldLat.git
   cd FieldLat
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

The core functionality is provided by the `generate_adaptive_lattice` function. Here is a minimal example of how to use it:

```python
import pyvista as pv
from fieldlat import generate_adaptive_lattice, load_febio_vtk

# 1. Load your mesh containing field data (e.g., 'stress')
# Ensure 'stress' is a scalar field in your VTK file.
mesh = load_febio_vtk('simulation_result.vtk', field_name='stress')

# 2. Generate the adaptive lattice
# - Low stress -> Large cells (Base Scale)
# - High stress -> Small cells (Dense Scale)
lattice = generate_adaptive_lattice(
    mesh,
    field_name='stress',
    lattice_type='gyroid', # Options: 'gyroid', 'diamond', 'primitive', 'lidinoid'
    structure_mode='sheet', # Options: 'sheet' (default) or 'strut'
    resolution=100,      # Grid resolution (higher = finer detail)
    base_scale=10.0,     # Frequency for low-stress areas
    dense_scale=25.0,    # Frequency for high-stress areas
    threshold=0.3        # Wall thickness constant (since we are in 'sheet' mode)
)

# 3. Save or Visualize
lattice.save('adaptive_lattice.stl')
lattice.plot(smooth_shading=True)
```

## Structure Modes: Sheet vs. Strut

FieldLat supports two primary generation modes:

- **Sheet (Shell) Mode**: Creates a hollow, double-walled surface following the zero-isosurface of the TPMS.
  - `structure_mode='sheet'`
  - `threshold`: Defines the **Wall Thickness** (must be > 0).

- **Strut (Network) Mode**: Creates a solid volume where the TPMS field is greater than the threshold. This results in a skeletal, network-like structure.
  - `structure_mode='strut'`
  - `threshold`: Defines the **Volume Fraction/Isovalue** (can be negative or positive, e.g., 0.0 for 50% density).

## Running the Demo

The repository includes a demo script that generates a dummy stress field on a cube and creates an adaptive lattice.

To run the demo:

```bash
python examples/demo_script.py
```

This will:
1. Create a dummy `input.vtk` file if one doesn't exist.
2. Generate an adaptive lattice (default: Gyroid) blending between two frequencies.
3. Display the result in an interactive 3D window.
4. Save the result to `output_lattice.stl`.

### Changing the Topology

To generate a different lattice structure (e.g., Lidinoid or Diamond), open `examples/demo_script.py` and modify the `lattice_type` parameter in the `generate_adaptive_lattice` function call:

```python
lattice = generate_adaptive_lattice(
    mesh, 
    field_name=field_name,
    lattice_type='lidinoid', # Change this to 'diamond', 'primitive', or 'lidinoid'
    resolution=60,
    # ... other parameters
)
```

## API Reference

### `generate_adaptive_lattice`

```python
def generate_adaptive_lattice(
    mesh: pv.DataSet,
    field_name: str,
    lattice_type: str = 'gyroid',
    structure_mode: str = 'sheet',
    resolution: int = 50,
    base_scale: float = 10.0,
    dense_scale: float = 25.0,
    threshold: float = 0.3,
    pad_width: int = 2
) -> pv.PolyData
```

**Parameters:**
- `mesh`: A PyVista DataSet containing the source geometry and scalar field.
- `field_name`: The name of the scalar array in `mesh.point_data` to use as the control field.
- `lattice_type`: The TPMS topology type. Options: `'gyroid'`, `'diamond'`, `'primitive'`, `'lidinoid'`.
- `structure_mode`: The generation mode. `'sheet'` (Shell) or `'strut'` (Skeletal). Defaults to `'sheet'`.
- `resolution`: The resolution of the voxel grid (cubed) used for marching cubes.
- `base_scale`: The frequency factor for regions with minimum field values (larger cells).
- `dense_scale`: The frequency factor for regions with maximum field values (smaller cells).
- `threshold`: Controls the isosurface level.
  - In **'sheet'** mode: Defines Wall Thickness (must be > 0).
  - In **'strut'** mode: Defines Volume Fraction/Isovalue.
- `pad_width`: Number of voxel layers to force to void at the boundaries to ensure a watertight mesh.

**Returns:**
- `pv.PolyData`: A mesh representing the generated lattice structure.
