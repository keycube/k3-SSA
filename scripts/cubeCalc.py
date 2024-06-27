import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calculate_angles(cuboid_center, cuboid_dims, pov):
    """Calculates the angles between the point of view and the faces of a cuboid.

    Parameters
    ----------
    cuboid_center : numpy.ndarray
        The center of the cuboid.
    cuboid_dims : numpy.ndarray
        The dimensions of the cuboid.
    pov : numpy.ndarray
        The point of view.

    Returns
    -------
    list : numpy.ndarray
        A list of booleans indicating whether each face is visible.
    list : numpy.ndarray
        A list of the visible surface areas of each face.
    list : numpy.ndarray
        A list of the outside angles of each face.
    int : int
        The number of visible faces.
    """

    half_dims = cuboid_dims / 2
    face_centers = {
        'front': np.array([cuboid_center[0], cuboid_center[1], cuboid_center[2] + half_dims[2]]),
        'back': np.array([cuboid_center[0], cuboid_center[1], cuboid_center[2] - half_dims[2]]),
        'left': np.array([cuboid_center[0] - half_dims[0], cuboid_center[1], cuboid_center[2]]),
        'right': np.array([cuboid_center[0] + half_dims[0], cuboid_center[1], cuboid_center[2]]),
        'top': np.array([cuboid_center[0], cuboid_center[1] + half_dims[1], cuboid_center[2]]),
        'bottom': np.array([cuboid_center[0], cuboid_center[1] - half_dims[1], cuboid_center[2]])
    }
    
    face_normals = {
        'front': np.array([0, 0, 1]),
        'back': np.array([0, 0, -1]),
        'left': np.array([-1, 0, 0]),
        'right': np.array([1, 0, 0]),
        'top': np.array([0, 1, 0]),
        'bottom': np.array([0, -1, 0])
    }

    visibilities = []
    visible_surface_areas = []
    outside_angles = []

    for face, center in face_centers.items():
        pov_to_face = center - pov
        pov_to_face_normalised = pov_to_face / np.linalg.norm(pov_to_face)
        face_normal = face_normals[face]
        dot_product = np.dot(pov_to_face_normalised, face_normal)
        visible = dot_product < 0
        visibilities.append(visible)
        if visible:
            incidence_angle = np.arccos(-dot_product)
            outside_angle = np.degrees(np.pi - incidence_angle)
            if face in ['front', 'back']:
                face_area = cuboid_dims[0] * cuboid_dims[1]
            elif face in ['left', 'right']:
                face_area = cuboid_dims[1] * cuboid_dims[2]
            else:
                face_area = cuboid_dims[0] * cuboid_dims[2]
            visible_surface_area = np.cos(incidence_angle) * face_area
        else:
            outside_angle = "Not Visible"
            visible_surface_area = "Not Visible"
        visible_surface_areas.append(visible_surface_area)
        outside_angles.append(outside_angle)

    num_visible_faces = sum(visibilities)
    return visibilities, visible_surface_areas, outside_angles, num_visible_faces

def generate_view_points(radius, step):
    """Generates view points on a sphere.

    Parameters
    ----------
    radius : float
        The radius of the sphere.
    step : int
        The step size in degrees.

    Returns
    -------
    numpy.ndarray
        The polar coordinates of the view points.
    numpy.ndarray
        The Cartesian coordinates of the view points.
    """

    phi = np.arange(0, 181, step)
    theta = np.arange(0, 361, step)

    phi, theta = np.meshgrid(phi, theta)
    phi = phi.flatten()
    theta = theta.flatten()

    x = radius * np.sin(np.radians(phi)) * np.cos(np.radians(theta))
    y = radius * np.sin(np.radians(phi)) * np.sin(np.radians(theta))
    z = radius * np.cos(np.radians(phi))

    return np.vstack((phi, theta)).T, np.vstack((x, y, z)).T

def test_angles(cuboid_center, cuboid_dims, radius, step):
    """Tests the angles and visibility of the faces of a cube from different points of view.

    Parameters
    ----------
    cuboid_center : numpy.ndarray
        The center of the cuboid.
    cuboid_dims : numpy.ndarray
        The dimensions of the cuboid.
    radius : float
        The radius of the sphere.
    step : int
        The step size in degrees.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the visibility index for each view point.
    """

    polar_points, view_points = generate_view_points(radius, step)
    results = []

    for idx, pov in enumerate(view_points):
        visibilities, visible_surface_areas, outside_angles, num_visible_faces = calculate_angles(cuboid_center, cuboid_dims, pov)
        total_visible_surface_area = sum(area if isinstance(area, (int, float)) else 0 for area in visible_surface_areas)
        results.append([polar_points[idx][0], polar_points[idx][1], total_visible_surface_area, num_visible_faces] + visible_surface_areas + outside_angles)

    columns = ['Phi (°)', 'Theta (°)', 'Total_Visible_Surface_Area', 'Num_Visible_Faces', 'Front_Surface_Area', 'Back_Surface_Area', 
               'Left_Surface_Area', 'Right_Surface_Area', 'Top_Surface_Area', 'Bottom_Surface_Area', 
               'Front_Outside_Angle', 'Back_Outside_Angle', 'Left_Outside_Angle', 'Right_Outside_Angle', 
               'Top_Outside_Angle', 'Bottom_Outside_Angle']
    df = pd.DataFrame(results, columns=columns)
    
    df['Normalised_Visible_Surface_Area'] = df['Total_Visible_Surface_Area'] / df['Total_Visible_Surface_Area'].max()
    df['Normalised_Num_Visible_Faces'] = df['Num_Visible_Faces'] / 6
    df['Visibility_Index'] = df['Normalised_Visible_Surface_Area'] + df['Normalised_Num_Visible_Faces']
    
    return df

def visualise_results(df, output_file):
    """Visualises the view points with the maximum visibility index.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the visibility index for each view point.
    output_file : str
        - The output file for the visualisation.
    """
    
    fig = plt.figure(figsize=(16, 8))
    
    ax1 = fig.add_subplot(121, projection='polar')
    colors = plt.cm.viridis((df['Visibility_Index'] - df['Visibility_Index'].min()) / 
                            (df['Visibility_Index'].max() - df['Visibility_Index'].min()))
    sc = ax1.scatter(np.radians(df['Theta (°)']), np.radians(df['Phi (°)']), c=colors, cmap='viridis', marker='o')
    plt.colorbar(sc, ax=ax1, label='Visibility Index')
    ax1.set_title('View Points with Maximum Visibility Index')
    
    ax2 = fig.add_subplot(122, projection='3d')
    x = 10 * np.sin(np.radians(df['Phi (°)'])) * np.cos(np.radians(df['Theta (°)']))
    y = 10 * np.sin(np.radians(df['Phi (°)'])) * np.sin(np.radians(df['Theta (°)']))
    z = 10 * np.cos(np.radians(df['Phi (°)']))
    sc = ax2.scatter(x, y, z, c=colors, cmap='viridis', marker='o')
    plt.colorbar(sc, ax=ax2, label='Visibility Index')
    ax2.set_title('3D View Points with Visibility Index')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    plt.savefig(output_file)
    plt.show()

def main():
    # Define the cuboid and sphere parameters
    cuboid_center = np.array([0, 0, 0])
    cuboid_dims = np.array([2, 2, 2])
    radius = 10
    step = 5

    # Test the angles and visibility of the faces of the cuboid
    results_df = test_angles(cuboid_center, cuboid_dims, radius, step)

    # Save the results to a CSV file
    csv_file = 'output.csv'
    results_df.to_csv(csv_file, index=False)

    # Visualise the results
    image_file = 'output.png'
    visualise_results(results_df, image_file)

    # Print the view points with the maximum visibility index
    max_visibility_index = results_df['Visibility_Index'].max()
    isolated_view_points = results_df[results_df['Visibility_Index'] == max_visibility_index]
    print(isolated_view_points)

if __name__ == '__main__':
    main()