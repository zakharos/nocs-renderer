import math
import util
import numpy as np
from scipy.linalg import expm, norm


class Icosahedron:
    def __init__(self, cfg):

        self.radius = util.read_cfg_float(cfg, 'sampling', 'icosahedron_radius', default=0.4)
        self.subdivisions = util.read_cfg_float(cfg, 'sampling', 'icosahedron_subdivision', default=2)
        self.skip_below = util.read_cfg_bool(cfg, 'sampling', 'skip_below', default=True)
        self.inplane = list(map(int, util.read_cfg_string(cfg, 'sampling', 'inplane', default="-45,46,15").split(',')))

    def __subdivide(self, verts, v1, v2, v3, level):
        # Once the final recursion level is reached, start adding verts (check for collisions)
        if level == 0:
            flag1, flag2, flag3 = False, False, False
            for v in verts:
                flag1 |= np.allclose(v, v1)
                flag2 |= np.allclose(v, v2)
                flag3 |= np.allclose(v, v3)
                if flag1 and flag2 and flag3:
                    break
            if not flag1:
                verts.append(v1)
            if not flag2:
                verts.append(v2)
            if not flag3:
                verts.append(v3)
            return

        # Create three new verts at the midpoints of each edge
        # and push vertices to the unit sphere
        v12 = (v1 + v2) / norm(v1 + v2)
        v23 = (v2 + v3) / norm(v2 + v3)
        v31 = (v1 + v3) / norm(v1 + v3)

        # Enter the recursion for 4 new faces
        self.__subdivide(verts, v1, v12, v31, level - 1)
        self.__subdivide(verts, v2, v23, v12, level - 1)
        self.__subdivide(verts, v3, v31, v23, level - 1)
        self.__subdivide(verts, v12, v23, v31, level - 1)

    def __create_viewpoints(self):
        # Generate the icosahedron
        x = 0.525731112119133606
        z = 0.850650808352039932
        # Vertices
        base = np.zeros((12, 3), np.float32)
        base[0] = [-x, 0, +z]
        base[1] = [+x, 0, +z]
        base[2] = [-x, 0, -z]
        base[3] = [+x, 0, -z]
        base[4] = [0, +z, +x]
        base[5] = [0, +z, -x]
        base[6] = [0, -z, +x]
        base[7] = [0, -z, -x]
        base[8] = [+z, +x, 0]
        base[9] = [-z, +x, 0]
        base[10] = [+z, -x, 0]
        base[11] = [-z, -x, 0]
        # Faces
        ind = [
            [0, 4, 1], [0, 9, 4], [9, 5, 4], [4, 5, 8], [4, 8, 1],
            [8, 10, 1], [8, 3, 10], [5, 3, 8], [5, 2, 3], [2, 7, 3],
            [7, 10, 3], [7, 6, 10], [7, 11, 6], [11, 0, 6], [0, 1, 6],
            [6, 1, 10], [9, 0, 11], [9, 11, 2], [9, 2, 5], [7, 2, 11]]

        verts = []

        # Enter the recursion to subdivide the icosahedron faces
        for i in ind:
            self.__subdivide(verts, base[i][0], base[i][1], base[i][2], self.subdivisions)

        return verts

    def __compute_rotation(self, eye):
        # Compute lookAt matrix for OpenCV
        up = [0, 0, 1]
        if eye[0] == 0 and eye[1] == 0 and eye[2] != 0:
            up = [-1, 0, 0]
        rot = np.zeros((3, 3))
        rot[:, 2] = -eye / norm(eye)
        rot[:, 0] = np.cross(rot[:, 2], up)
        rot[:, 0] /= norm(rot[:, 0])
        rot[:, 1] = np.cross(rot[:, 2], rot[:, 0])

        return rot.T

    def create_poses(self):
        viewpoints = self.__create_viewpoints()

        poses = []
        for vertex in viewpoints:
            for angle in range(self.inplane[0], self.inplane[1], self.inplane[2]):
                rot = self.__compute_rotation(vertex)
                pose = np.eye(4)
                # Add in-plane rotation using Rodrigues' formula: R = exp(Ab)
                rodriguez = np.asarray([0, 0, 1]) * (angle * math.pi / 180.0)
                angle_axis = expm(np.cross(np.eye(3), rodriguez))
                pose[0:3, 0:3] = angle_axis @ rot
                pose[0:3, 3] = [0, 0, self.radius]  # scale = distance from the object

                if self.skip_below and vertex[2] < 0:
                    continue

                poses.append(pose)
        return poses
