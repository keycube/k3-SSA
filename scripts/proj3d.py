# https://skannai.medium.com/projecting-3d-points-into-a-2d-screen-58db65609f24
# Projecting 3D Points into a 2D Screen

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def orthographic_projection(left, right, bottom, top, near, far):
    """
    Create an orthographic projection matrix

    Parameters
    ----------
    left : float
        Left coordinate of the view volume
    right : float
        Right coordinate of the view volume
    bottom : float
        Bottom coordinate of the view volume
    top : float
        Top coordinate of the view volume
    near : float
        Near coordinate of the view volume
    far : float
        Far coordinate of the view volume

    Returns
    -------
    np.ndarray
        Orthographic projection matrix
    """

    op_matrix = np.array([
        [2 / (right - left), 0, 0, -(right + left) / (right - left)],
        [0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
        [0, 0, -2 / (far - near), -(far + near) / (far - near)],
        [0, 0, 0, 1]
    ])

    return op_matrix

def perspective_projection(left, right, bottom, top, near, far):
    """
    Create a perspective projection matrix

    Parameters
    ----------
    left : float
        Left coordinate of the view volume
    right : float
        Right coordinate of the view volume
    bottom : float
        Bottom coordinate of the view volume
    top : float
        Top coordinate of the view volume
    near : float
        Near coordinate of the view volume
    far : float
        Far coordinate of the view volume

    Returns
    -------
    np.ndarray
        Perspective projection matrix
    """

    pp_matrix = np.array([
        [(2 * near) / (right - left), 0, (right + left) / (right - left), 0],
        [0, (2 * near) / (top - bottom), (top + bottom) / (top - bottom), 0],
        [0, 0, -(far + near) / (far - near), -(2 * far * near) / (far - near)],
        [0, 0, -1, 0]
    ])

    return pp_matrix

def viewport(nx, ny):
    """
    Create a viewport matrix

    Parameters
    ----------
    nx : int
        Width of the viewport
    ny : int
        Height of the viewport

    Returns
    -------
    np.ndarray
        Viewport matrix
    """

    return np.array([
        [nx / 2, 0, 0, (nx - 1) / 2],
        [0, ny / 2, 0, (ny - 1) / 2],
        [0, 0, 0.5, 0.5],
    ])

def project_points(points, camera_matrix, projection_matrix, viewport_matrix):
    """
    Project 3D points onto a 2D screen

    Parameters
    ----------
    points : np.ndarray
        3D points to project
    camera_matrix : np.ndarray
        Camera matrix
    projection_matrix : np.ndarray
        Projection matrix
    viewport_matrix : np.ndarray
        Viewport matrix

    Returns
    -------
    np.ndarray
        2D points on the screen
    """

    points_after_CM = camera_matrix @ points
    points_after_PM = projection_matrix @ points_after_CM
    points_after_PM /= points_after_PM[3]
    points_after_VP = viewport_matrix @ points_after_PM

    return points_after_VP

def plot_cube(ax3d, ax2d, cube_vertices, cube_edges, projection, camera_matrix, viewport_matrix):
    """
    Plot a cube in 3D and 2D

    Parameters
    ----------
    ax3d : matplotlib.axes._subplots.Axes3DSubplot
        3D subplot
    ax2d : matplotlib.axes._subplots.AxesSubplot
        2D subplot
    cube_vertices : np.ndarray
        Vertices of the cube
    cube_edges : list
        Edges of the cube
    projection : np.ndarray
        Projection matrix
    camera_matrix : np.ndarray
        Camera matrix
    viewport_matrix : np.ndarray
        Viewport matrix
    """

    for edge in cube_edges:
        ax3d.plot(cube_vertices[edge, 0], cube_vertices[edge, 1], cube_vertices[edge, 2], color='blue')

        cube_after_projection = project_points(cube_vertices.T, camera_matrix, projection, viewport_matrix)
        for edge in cube_edges:
            start_idx, end_idx = edge
            start_point = cube_after_projection[:2, start_idx]
            end_point = cube_after_projection[:2, end_idx]
            ax2d.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='red')

def calculate_visible_surface_percentage(cube_vertices, cube_edges, projection_matrix, camera_matrix):
    """
    Calculate the percentage of the object's surface that is visible.

    Parameters:
    cube_vertices : np.ndarray
        Vertices of the cube
    cube_edges : list
        Edges of the cube
    projection_matrix : np.ndarray
        Projection matrix
    camera_matrix : np.ndarray
        Camera matrix

    Returns:
    float
        Percentage of the object's surface that is visible
    """
    # Project cube vertices
    cube_after_projection = project_points(cube_vertices.T, camera_matrix, projection_matrix, np.eye(4))

    # Calculate visible surface area
    visible_surface_area = 0
    for edge in cube_edges:
        start_idx, end_idx = edge
        start_point = cube_after_projection[:2, start_idx]
        end_point = cube_after_projection[:2, end_idx]
        if np.all(start_point >= 0) or np.all(end_point >= 0):
            # Calculate area of face using cross product
            v1 = np.append(cube_after_projection[:2, start_idx], 1)
            v2 = np.append(cube_after_projection[:2, end_idx], 1)
            visible_surface_area += np.abs(np.cross(v1, v2))

    # Calculate total surface area
    total_surface_area = 12 * np.linalg.norm(cube_vertices[0] - cube_vertices[1])**2

    # Calculate percentage of visible surface
    visible_surface_percentage = (visible_surface_area / total_surface_area) * 100
    return visible_surface_percentage

def calculate_object_coverage(cube_vertices, projection_matrix, camera_matrix, viewport_matrix):
    """
    Calculate the percentage occupied by the object in the field of view.

    Parameters:
    cube_vertices : np.ndarray
        Vertices of the cube
    projection_matrix : np.ndarray
        Projection matrix
    camera_matrix : np.ndarray
        Camera matrix
    viewport_matrix : np.ndarray
        Viewport matrix

    Returns:
    float
        Percentage occupied by the object in the field of view
    """
    # Project cube vertices
    cube_after_projection = project_points(cube_vertices.T, camera_matrix, projection_matrix, viewport_matrix)

    # Calculate bounding box of the object in the viewport
    min_x = np.min(cube_after_projection[0])
    max_x = np.max(cube_after_projection[0])
    min_y = np.min(cube_after_projection[1])
    max_y = np.max(cube_after_projection[1])

    # Calculate area occupied by object
    object_width = max_x - min_x
    object_height = max_y - min_y
    object_area = max(object_width, 0) * max(object_height, 0)

    # Calculate total viewport area
    viewport_width = viewport_matrix[0, 0]
    viewport_height = viewport_matrix[1, 1]
    total_viewport_area = viewport_width * viewport_height

    # Calculate percentage occupied by object
    object_coverage_percentage = (object_area / total_viewport_area) * 100
    return object_coverage_percentage

def main():
    # Homogeneous point we want to convert
    point_3d = np.array([2, 2, -10, 1])

    # Type of the projection we want
    projection_type = 'perspective'  # 'orthographic'

    # Coordinates of the view volume
    left, right = -3, 3
    bottom, top = -3, 3
    near, far = 5, 20

    # Creating camera matrix
    rotation_matrix = np.eye(4)  # Identity matrix, no rotation
    translation_matrix = np.array([
        [1, 0, 0, 2],
        [0, 1, 0, 2],
        [0, 0, 1, 2],
        [0, 0, 0, 1]
    ])
    camera_matrix = rotation_matrix @ translation_matrix

    #ViewPort Matrix
    nx = 600
    ny = 600
    viewport_matrix = viewport(nx, ny)

    # Choosing projection matrix associated with projection type
    if projection_type == 'orthographic':
        projection_matrix = orthographic_projection(left, right, bottom, top, near, far)
    elif projection_type == 'perspective':
        projection_matrix = perspective_projection(left, right, bottom, top, near, far)

    # Create cube vertices and edges
    cube_vertices = np.array([
        [-1, -1, -1, 1], [1, -1, -1, 1], [1, 1, -1, 1], [-1, 1, -1, 1],
        [-1, -1, 1, 1], [1, -1, 1, 1], [1, 1, 1, 1], [-1, 1, 1, 1]
    ])

    # Translate cube vertices to center at (0, 0, -10)
    translation_vector = np.array([0, 0, -10, 0])
    cube_vertices += translation_vector
    cube_edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]

    # Create a figure and 3D subplot
    fig = plt.figure(figsize=(10, 6))
    ax3d = fig.add_subplot(121, projection='3d')
    ax2d = fig.add_subplot(122)

    # Plot the cube in 3D and 2D
    plot_cube(ax3d, ax2d, cube_vertices, cube_edges, projection_matrix, camera_matrix, viewport_matrix)

    # Set labels and title
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.set_title('3D Cube Projection')
    ax2d.set_xlabel('X')
    ax2d.set_ylabel('Y')
    ax2d.set_xlim(0, 600)
    ax2d.set_ylim(0, 600)
    ax2d.set_title('2D Projection on Screen')

    plt.tight_layout()
    plt.show()

    # Calculate percentage of visible surface
    visible_surface_percentage = calculate_visible_surface_percentage(cube_vertices, cube_edges, projection_matrix, camera_matrix)
    print("Percentage of visible surface:", visible_surface_percentage)

    # Calculate percentage occupied by the object in the field of view
    object_coverage_percentage = calculate_object_coverage(cube_vertices, projection_matrix, camera_matrix, viewport_matrix)
    print("Percentage occupied by object in field of view:", object_coverage_percentage)

if __name__ == "__main__":
    main()