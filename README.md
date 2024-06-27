# k3-ssa

Keycube performance study against Shoulder Surfing Attacks (SSA)

# 3D to 2D Projection and Analysis

- File -> [scripts/proj3d](scripts/proj3d.py)

This script provides tools for projecting 3D points onto a 2D screen using either orthographic or perspective projection. It also includes visualization of a 3D cube in both 3D and 2D views, and calculations of the percentage of visible surface area and object coverage in the viewport.

More information can be found using the following [link](https://skannai.medium.com/projecting-3d-points-into-a-2d-screen-58db65609f24).

# Cube Visibility Analysis

- File -> [scripts/cubeCalc](scripts/cubeCalc.py)

This project provides a comprehensive analysis of the visibility and angles of the faces of a cube from various points of view. It includes generating view points on a sphere, calculating visibility, and visualizing the results.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Functions](#functions)
- [Examples](#examples)
- [Results](#results)

## Introduction

This script calculates the visibility and angles of the faces of a cube from a given point of view and visualizes the results. It generates view points on a sphere around the cube and evaluates which faces are visible from each view point.

## Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib

## Installation

To install the required packages, run:

```bash
pip install numpy pandas matplotlib
```

## Usage

To run the script, simply execute:

```bash
python scripts/cubeCalc.py
```

## Functions

### `calculate_angles(cube_center, cube_size, pov)`

Calculates the angles and visibility of the faces of a cube from a given point of view.

**Parameters:**

- `cube_center` : np.ndarray
  - The center of the cube.
- `cube_size` : float
  - The size of the cube.
- `pov` : np.ndarray
  - The point of view.

**Returns:**

- `list`: A list of booleans indicating the visibility of each face.
- `list`: A list of the visible surface areas of each face.
- `list`: A list of the outside angles of each face.
- `int`: The number of visible faces.

### `generate_view_points(radius, step)`

Generates view points on a sphere.

**Parameters:**

- `radius` : float
  - The radius of the sphere.
- `step` : float
  - The step size for generating view points.

**Returns:**

- `np.ndarray`: An array of polar coordinates.
- `np.ndarray`: An array of view points.

### `test_angles(cube_center, cube_size, radius, step)`

Tests the angles and visibility of the faces of a cube from different points of view.

**Parameters:**

- `cube_center` : np.ndarray
  - The center of the cube.
- `cube_size` : float
  - The size of the cube.
- `radius` : float
  - The radius of the sphere.
- `step` : float
  - The step size for generating view points.

**Returns:**

- `pd.DataFrame`: A DataFrame containing the results of the tests.

### `visualize_results(df, output_file)`

Visualizes the results of the tests.

**Parameters:**

- `df` : pd.DataFrame
  - A DataFrame containing the results of the tests.
- `output_file` : str
  - The output file for the visualization.

## Examples

To run the main analysis and generate the results:

```python
cube_center = np.array([0, 0, 0])
cube_size = 2
radius = 10
step = 5

results_df = test_angles(cube_center, cube_size, radius, step)
results_df.to_csv('output.csv', index=False)
visualize_results(results_df, 'output.png')
```

## Results

The results include a CSV file `output.csv` containing the visibility and angle calculations, and a visualization saved as `output.png`. The visualization shows the view points with maximum visibility index on a polar and 3D plot.
