Simple renderer to output RGB, Depth, and NOCS correspondences. Based on the implementation of Wadim Kehl: [https://github.com/wadimkehl/ssd-6d/tree/master/rendering](https://github.com/wadimkehl/ssd-6d/tree/master/rendering).

## Installation
* install anaconda or miniconda
* set up a virtual environment using: "conda env create -n renderer -f env.yml"
* activate a virtual environment using: source activate renderer

## Usage
To run the code activate the created virtual environment and execute the following command:
"python main.py -c config.ini", where c is a config file

## Datasets
An example 3D model taken from the [HomebrewedDB dataset](http://campar.in.tum.de/personal/ilic/homebreweddb/index.html) is located in the *db* folder.

### Config file:
* icosahedron_radius [m]
* icosahedron_subdivision - number of icosahedron subdivisions 
* skip_below - hemisphere (True) vs fullsphere (False) for rendering
* inplane - inplane rotation angles for each vertex [from, to, stride]
* resolution of the output image [px]
* intrinsic parameters of the camera:  fx, cx, fy, cy
* type: d, rgb, rgbd, normals, corr (correspondences)
