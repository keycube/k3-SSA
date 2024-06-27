# k3-ssa

Keycube performance study against Shoulder Surfing Attacks (SSA)

# 3D to 2D Projection and Analysis

- File -> [scripts/proj3d](scripts/proj3d.py)

This script provides tools for projecting 3D points onto a 2D screen using either orthographic or perspective projection. It also includes visualisation of a 3D cube in both 3D and 2D views, and calculations of the percentage of visible surface area and object coverage in the viewport.

More information can be found using the following [link](https://skannai.medium.com/projecting-3d-points-into-a-2d-screen-58db65609f24).

# Cube Visibility Analysis

- File -> [scripts/cubeCalc](scripts/cubeCalc.py)

This project provides a comprehensive analysis of the visibility and angles of the faces of a cube from various points of view. It includes generating view points on a sphere, calculating visibility, and visualising the results.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Functions](#functions)
- [Examples](#examples)
- [Results](#results)

## Introduction

This script calculates the visibility and angles of the faces of a cube from a given point of view and visualises the results. It generates view points on a sphere around the cube and evaluates which faces are visible from each view point.

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

### `calculate_angles(cuboid_center, cuboid_dims, pov)`

Calculates the angles between the point of view and the faces of a cuboid.

**Parameters:**

- `cuboid_center`: numpy.ndarray
  - The center of the cuboid.
- `cuboid_dims`: numpy.ndarray
  - The dimensions of the cuboid.
- `pov`: numpy.ndarray
  - The point of view.

**Returns:**

- `list`: numpy.ndarray
  - A list of booleans indicating whether each face is visible.
- `list`: numpy.ndarray
  - A list of the visible surface areas of each face.
- `list`: numpy.ndarray
  - A list of the outside angles of each face.
- `int`: int
  - The number of visible faces.

### `generate_view_points(radius, step)`

Generates view points on a sphere.

**Parameters:**

- `radius` : float
  - The radius of the sphere.
- `step` : float
  - The step size in degrees.

**Returns:**

- `np.ndarray`: The polar coordinates of the view points.
- `np.ndarray`: The Cartesian coordinates of the view points.

### `test_angles(cuboid_center, cuboid_dims, radius, step)`

Tests the angles and visibility of the faces of a cube from different points of view.

**Parameters:**

- `cuboid_center`: numpy.ndarray
  - The center of the cuboid.
- `cuboid_dims`: numpy.ndarray
  - The dimensions of the cuboid.
- `radius`: float
  - The radius of the sphere.
- `step`: int
  - The step size in degrees.

**Returns:**

- `pd.DataFrame`: A DataFrame containing the visibility index for each view point.

### `visualise_results(df, output_file)`

Visualises the view points with the maximum visibility index.

**Parameters:**

- `df` : pd.DataFrame
  - The DataFrame containing the visibility index for each view point.
- `output_file` : str
  - The output file for the visualisation.

## Examples

To run the main analysis and generate the results:

```python
  cuboid_center = np.array([0, 0, 0])
  cuboid_dims = np.array([2, 2, 2])
  radius = 10
  step = 5
```

## Results

The results include a CSV file `output.csv` containing the visibility and angle calculations, and a visualisation saved as `output.png`. The visualisation shows the view points with maximum visibility index on a polar and 3D plot.
