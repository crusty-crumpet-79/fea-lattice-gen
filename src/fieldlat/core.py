import numpy as np
import pyvista as pv
from skimage.measure import marching_cubes

def get_lattice_field(lattice_type: str, scale: float, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Computes the scalar field values for the specified lattice topology.
    
    Args:
        lattice_type (str): 'gyroid', 'diamond', 'primitive', or 'lidinoid'.
        scale (float): The frequency/scale factor for the coordinates.
        X, Y, Z (np.ndarray): Meshgrid coordinates.
        
    Returns:
        np.ndarray: The computed scalar field.
    """
    if lattice_type == 'lidinoid':
        # Note: Lidinoid uses 2*x in its formula, effectively doubling the density.
        # We divide by 2.0 here so that 'base_scale' results in roughly the same cell size
        # as other lattice types.
        sx, sy, sz = (scale * X) / 2.0, (scale * Y) / 2.0, (scale * Z) / 2.0
        
        term1 = 0.5 * (np.sin(2*sx)*np.cos(sy)*np.sin(sz) + 
                       np.sin(2*sy)*np.cos(sz)*np.sin(sx) + 
                       np.sin(2*sz)*np.cos(sx)*np.sin(sy))
        
        term2 = 0.5 * (np.cos(2*sx)*np.cos(2*sy) + 
                       np.cos(2*sy)*np.cos(2*sz) + 
                       np.cos(2*sz)*np.cos(2*sx))
                       
        return term1 - term2 + 0.15

    # Standard scaling for other types
    sx, sy, sz = scale * X, scale * Y, scale * Z

    if lattice_type == 'gyroid':
        return np.sin(sx) * np.cos(sy) + np.sin(sy) * np.cos(sz) + np.sin(sz) * np.cos(sx)

    elif lattice_type == 'diamond':
        # Schwarz D
        return (np.sin(sx) * np.sin(sy) * np.sin(sz) + 
                np.sin(sx) * np.cos(sy) * np.cos(sz) + 
                np.cos(sx) * np.sin(sy) * np.cos(sz) + 
                np.cos(sx) * np.cos(sy) * np.sin(sz))

    elif lattice_type == 'primitive':
        # Schwarz P
        return np.cos(sx) + np.cos(sy) + np.cos(sz)

    else:
        raise ValueError(f"Unsupported lattice_type: '{lattice_type}'. "
                         "Choose from 'gyroid', 'diamond', 'primitive', 'lidinoid'.")

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
) -> pv.PolyData:
    """
    Generates an adaptive lattice (TPMS) with variable cell size via lattice blending.

    Args:
        mesh (pv.DataSet): The source mesh containing the scalar field.
        field_name (str): The name of the scalar field in point_data.
        lattice_type (str): Topology type ('gyroid', 'diamond', 'primitive', 'lidinoid').
        structure_mode (str): 'sheet' (Shell) or 'strut' (Skeletal). 
                              Defaults to 'sheet'.
        resolution (int): Resolution of the voxel grid (cubed).
        base_scale (float): The frequency for low-stress areas (larger cells).
        dense_scale (float): The frequency for high-stress areas (smaller cells).
        threshold (float): Controls density/thickness.
                           - In 'sheet' mode: Defines Wall Thickness (must be > 0).
                           - In 'strut' mode: Defines Volume Fraction/Isovalue.
        pad_width (int): Number of voxel layers to force to void at the boundaries.
                         This ensures the mesh is watertight (capped). Default is 2.

    Returns:
        pv.PolyData: The extracted isosurface mesh of the lattice.
    """
    
    # 1. Define the Bounding Box and Voxel Grid
    bounds = mesh.bounds
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[2], bounds[3], resolution)
    z = np.linspace(bounds[4], bounds[5], resolution)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # 2. Interpolate the Field Data onto the Voxel Grid
    grid = pv.ImageData(
        dimensions=(resolution, resolution, resolution),
        spacing=(
            (bounds[1] - bounds[0]) / (resolution - 1),
            (bounds[3] - bounds[2]) / (resolution - 1),
            (bounds[5] - bounds[4]) / (resolution - 1),
        ),
        origin=(bounds[0], bounds[2], bounds[4])
    )
    
    sampled_grid = grid.sample(mesh)
    field_values = sampled_grid.point_data[field_name].reshape((resolution, resolution, resolution), order='F')

    if np.any(np.isnan(field_values)):
        field_values = np.nan_to_num(field_values, nan=np.nanmin(field_values))

    # 3. Normalize Field to Weight Map [0.0, 1.0]
    f_min, f_max = field_values.min(), field_values.max()
    if f_max > f_min:
        w = (field_values - f_min) / (f_max - f_min)
    else:
        w = np.zeros_like(field_values)

    # 4. Calculate Blended Lattice Field
    field_low = get_lattice_field(lattice_type, base_scale, X, Y, Z)
    field_high = get_lattice_field(lattice_type, dense_scale, X, Y, Z)

    # Linear interpolation between low and high frequency fields
    result = (field_low * (1 - w)) + (field_high * w)

    # 5. Apply Structure Mode Logic
    if structure_mode == 'sheet':
        # Sheet/Shell: Wall around the zero isosurface
        # Values < 0 are inside the wall (solid), Values > 0 are air (void)
        scalar_field = np.abs(result) - threshold
    elif structure_mode == 'strut':
        # Strut/Network: Solidify one domain
        # Using the requested formula: blended_lattice - threshold
        scalar_field = result - threshold
    else:
        raise ValueError(f"Unknown structure_mode: '{structure_mode}'. Use 'sheet' or 'strut'.")

    # 6. Apply Padding to force watertight mesh
    # We force the boundary voxels to a positive value (Void) to close the mesh.
    if pad_width > 0:
        scalar_field[:pad_width, :, :] = 1.0
        scalar_field[-pad_width:, :, :] = 1.0
        scalar_field[:, :pad_width, :] = 1.0
        scalar_field[:, -pad_width:, :] = 1.0
        scalar_field[:, :, :pad_width] = 1.0
        scalar_field[:, :, -pad_width:] = 1.0

    # 7. Extract Surface using Marching Cubes
    verts, faces, normals, values = marching_cubes(
        scalar_field, 
        level=0.0, 
        spacing=grid.spacing
    )

    # Offset vertices by the origin
    verts += np.array([bounds[0], bounds[2], bounds[4]])

    # 8. Convert to PyVista Mesh
    n_faces = faces.shape[0]
    padding = np.full((n_faces, 1), 3)
    pv_faces = np.hstack((padding, faces)).flatten()

    surface_mesh = pv.PolyData(verts, pv_faces)
    
    return surface_mesh
