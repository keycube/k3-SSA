import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calculate_angles(cube_center, cube_size, pov):
    """Calculates the angles and visibility of the faces of a cube from a given point of view

    Parameters
    ----------
    cube_center : np.ndarray
        The center of the cube
    cube_size : float
        The size of the cube
    pov : np.ndarray
        The point of view

    Returns
    -------
    list
        A list of booleans indicating the visibility of each face
    list
        A list of the visible surface areas of each face
    list
        A list of the outside angles of each face
    int
        The number of visible faces
    """

    half_size = cube_size / 2
    face_centers = {
        'front': np.array([cube_center[0], cube_center[1], cube_center[2] + half_size]),
        'back': np.array([cube_center[0], cube_center[1], cube_center[2] - half_size]),
        'left': np.array([cube_center[0] - half_size, cube_center[1], cube_center[2]]),
        'right': np.array([cube_center[0] + half_size, cube_center[1], cube_center[2]]),
        'top': np.array([cube_center[0], cube_center[1] + half_size, cube_center[2]]),
        'bottom': np.array([cube_center[0], cube_center[1] - half_size, cube_center[2]])
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
        pov_to_face_normalized = pov_to_face / np.linalg.norm(pov_to_face)
        face_normal = face_normals[face]
        dot_product = np.dot(pov_to_face_normalized, face_normal)
        visible = dot_product < 0
        visibilities.append(visible)
        if visible:
            incidence_angle = np.arccos(-dot_product)
            outside_angle = np.degrees(np.pi - incidence_angle)
            visible_surface_area = np.cos(incidence_angle) * cube_size**2
        else:
            outside_angle = "Not Visible"
            visible_surface_area = "Not Visible"
        visible_surface_areas.append(visible_surface_area)
        outside_angles.append(outside_angle)

    num_visible_faces = sum(visibilities)
    return visibilities, visible_surface_areas, outside_angles, num_visible_faces

def generate_view_points(radius, step):
    """Generates view points on a sphere

    Parameters
    ----------
    radius : float
        The radius of the sphere
    step : float
        The step size for generating view points

    Returns
    -------
    np.ndarray
        An array of polar coordinates
    np.ndarray
        An array of view points
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

def test_angles(cube_center, cube_size, radius, step):
    """Tests the angles and visibility of the faces of a cube from different points of view

    Parameters
    ----------
    cube_center : np.ndarray
        The center of the cube
    cube_size : float
        The size of the cube
    radius : float
        The radius of the sphere
    step : float
        The step size for generating view points

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the results of the tests
    """

    polar_points, view_points = generate_view_points(radius, step)
    results = []

    for idx, pov in enumerate(view_points):
        visibilities, visible_surface_areas, outside_angles, num_visible_faces = calculate_angles(cube_center, cube_size, pov)
        total_visible_surface_area = sum(area if isinstance(area, (int, float)) else 0 for area in visible_surface_areas)
        results.append([polar_points[idx][0], polar_points[idx][1], total_visible_surface_area, num_visible_faces] + visible_surface_areas + outside_angles)

    columns = ['Phi (°)', 'Theta (°)', 'Total_Visible_Surface_Area', 'Num_Visible_Faces', 'Front_Surface_Area', 'Back_Surface_Area', 
               'Left_Surface_Area', 'Right_Surface_Area', 'Top_Surface_Area', 'Bottom_Surface_Area', 
               'Front_Outside_Angle', 'Back_Outside_Angle', 'Left_Outside_Angle', 'Right_Outside_Angle', 
               'Top_Outside_Angle', 'Bottom_Outside_Angle']
    df = pd.DataFrame(results, columns=columns)
    
    df['Normalized_Visible_Surface_Area'] = df['Total_Visible_Surface_Area'] / df['Total_Visible_Surface_Area'].max()
    df['Normalized_Num_Visible_Faces'] = df['Num_Visible_Faces'] / 6
    df['Visibility_Index'] = df['Normalized_Visible_Surface_Area'] + df['Normalized_Num_Visible_Faces']
    
    return df

def visualize_results(df, output_file):
    """Visualizes the results of the tests

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the results of the tests
    output_file : str
        The output file for the visualization
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
    # Parse command line arguments
    cube_center = np.array([0, 0, 0])
    cube_size = 2
    radius = 10
    step = 5

    results_df = test_angles(cube_center, cube_size, radius, step)

    # Save results to a CSV file
    csv_file = 'output.csv'
    results_df.to_csv(csv_file, index=False)

    image_file = 'output.png'
    visualize_results(results_df, image_file)

    # Print isolated view points with maximum visibility index
    max_visibility_index = results_df['Visibility_Index'].max()
    isolated_view_points = results_df[results_df['Visibility_Index'] == max_visibility_index]
    print(isolated_view_points)

if __name__ == "__main__":
    main()