import numpy as np
import vispy
import plyfile
import util


class Model:
    def __init__(self):
        self.vertices = None
        self.indices = None
        self.colors = None
        self.normals = None
        self.collated = None
        self.vertex_buffer = None
        self.index_buffer = None
        self.bb = None
        self.diameter = None

    def load(self, model, color='rgb', scale='mm'):
        ply = plyfile.PlyData.read(model)
        self.vertices = np.zeros((ply['vertex'].count, 3))

        # Define the scale factor
        if scale == 'mm':
            scale_factor = 1000
        else:
            scale_factor = 1

        self.vertices[:, 0] = np.array(ply['vertex']['x'])/scale_factor
        self.vertices[:, 1] = np.array(ply['vertex']['y'])/scale_factor
        self.vertices[:, 2] = np.array(ply['vertex']['z'])/scale_factor

        self.normals = np.zeros((ply['vertex'].count, 3))
        self.normals[:, 0] = np.array(ply['vertex']['nx'])
        self.normals[:, 1] = np.array(ply['vertex']['ny'])
        self.normals[:, 2] = np.array(ply['vertex']['nz'])
        # self.normals = np.zeros((ply['vertex'].count, 3))
        # self.normals[:, 0] = np.array(ply['vertex']['blue'])
        # self.normals[:, 1] = np.array(ply['vertex']['green'])
        # self.normals[:, 2] = np.array(ply['vertex']['red'])

        self.colors = np.zeros((ply['vertex'].count, 3))
        self.colors[:, 0] = np.array(ply['vertex']['blue'])
        self.colors[:, 1] = np.array(ply['vertex']['green'])
        self.colors[:, 2] = np.array(ply['vertex']['red'])
        self.colors /= 255.0

        if color == 'corr':
            self.colors[:, 0] = util.normalize(np.array(ply['vertex']['x']))
            self.colors[:, 1] = util.normalize(np.array(ply['vertex']['y']))
            self.colors[:, 2] = util.normalize(np.array(ply['vertex']['z']))

        self.indices = np.asarray(list(ply['face']['vertex_indices']))
        vertices_type = [('a_position', np.float32, 3), ('a_color', np.float32, 3), ('a_normal', np.float32, 3)]
        self.collated = np.asarray(list(zip(self.vertices, self.colors, self.normals)), vertices_type)

        # Create buffers
        self.vertex_buffer = vispy.gloo.VertexBuffer(self.collated)
        self.index_buffer = vispy.gloo.IndexBuffer(self.indices.flatten().astype(np.uint32))
