import numpy as np
import pyvista as pv
from skimage.measure import marching_cubes

def generate_adaptive_lattice(
    mesh: pv.DataSet,
    field_name: str,
    resolution: int = 50,
    base_scale: float = 10.0,
    dense_scale: float = 25.0,
    threshold: float = 0.3,
    pad_width: int = 2
) -> pv.PolyData:
    """
    Generates an adaptive Gyroid lattice with variable cell size via lattice blending.

    Args:
        mesh (pv.DataSet): The source mesh containing the scalar field.
        field_name (str): The name of the scalar field in point_data.
        resolution (int): Resolution of the voxel grid (cubed).
        base_scale (float): The frequency for low-stress areas (larger cells).
        dense_scale (float): The frequency for high-stress areas (smaller cells).
        threshold (float): Constant wall thickness threshold.
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

    # 4. Calculate Blended Gyroid Field
    # gyroid(x, y, z) = sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x)
    
    def get_gyroid(scale, X, Y, Z):
        # We scale the coordinates by the frequency
        sx, sy, sz = scale * X, scale * Y, scale * Z
        return np.sin(sx) * np.cos(sy) + np.sin(sy) * np.cos(sz) + np.sin(sz) * np.cos(sx)

    gyroid_low = get_gyroid(base_scale, X, Y, Z)
    gyroid_high = get_gyroid(dense_scale, X, Y, Z)

    # Linear interpolation between low and high frequency fields
    result = (gyroid_low * (1 - w)) + (gyroid_high * w)

    # 5. Apply Wall Thickness Threshold
    # Isosurface: |result| - threshold = 0
    # Values < 0 are inside the wall (solid), Values > 0 are air (void)
    scalar_field = np.abs(result) - threshold

    # 6. Apply Padding to force watertight mesh
    # We force the boundary voxels to a positive value (Void) to close the mesh.
    if pad_width > 0:
        # Create a boolean mask of the same shape, initially all False (keep values)
        # Or simpler, just set the slices.
        # We set to 1.0 which is > 0 (Void).
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