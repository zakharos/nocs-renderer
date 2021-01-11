import numpy as np
from vispy import app, gloo
from glumpy import gl

import util

app.use_app('PyGlet')

_vertex_code = """
uniform mat4 u_mv;
uniform mat4 u_mvp;
uniform vec3 u_light_eye_pos;
//uniform int u_filter;

attribute vec3 a_position;
attribute vec3 a_normal;
attribute vec3 a_color;
varying vec4 v_color;
varying vec3 v_eye_pos;
varying vec3 v_L;
varying vec3 v_normal;

void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
    v_color = vec4(a_color, 1.0);
    //v_color = vec4(1.0, 1.0, 1.0, 0.8);
    v_eye_pos = (u_mv * vec4(a_position, 1.0)).xyz; // Vertex position in eye coordinates
    v_L = normalize(u_light_eye_pos - v_eye_pos); // Vector to the light
    v_normal = (u_mv * vec4(a_normal, 0.0)).xyz;
    
    //float dot_prod = dot(normalize(v_normal), normalize(v_eye_pos));
    //bool u_filter = true;
    //if (u_filter != 0  && dot_prod > 0) v_color = vec4(0, 0, 0, 1);
    //if (u_filter != 0  && dot_prod < -0.5) v_color = vec4(0, 0, 0, 1);

}
"""

_fragment_code_normals = """
uniform float u_light_ambient_w;
uniform vec3 light_color;
varying vec3 v_color;
varying vec3 v_eye_pos;
varying vec3 v_L;
varying vec3 v_normal;

void main() {
    // Face normal in eye coordinates
    vec3 face_normal = normalize(cross(dFdx(v_eye_pos), dFdy(v_eye_pos)));
    // float dot_prod = dot(face_normal, v_eye_pos);
    // int w = 1;
    // if((dot_prod) < 0.0) w = 0;
    //gl_FragColor = vec4((face_normal + 1.) / 2., 1.0);

    gl_FragColor =  vec4((v_normal + 1.) / 2., 1.0);
    
}
"""

_fragment_code_rgb = """
varying vec4 v_color;

void main() {
    gl_FragColor = v_color;
}
"""


class Renderer(app.Canvas):
    def __init__(self, cfg, type='rgbd', clip_near=0.01, clip_far=10.0):
        size = util.read_cfg_tuple(cfg, 'renderer', 'resolution', default=(640, 480))
        app.Canvas.__init__(self, config=dict(samples=8), show=False, size=size)
        self.shape = (size[1], size[0])
        self.clip_near = clip_near
        self.clip_far = clip_far
        self.yz_flip = np.eye(4, dtype=np.float32)
        self.yz_flip[1, 1], self.yz_flip[2, 2] = -1, -1
        self.cam = util.read_cfg_cam(cfg, 'renderer', 'intrinsics', default=np.identity(3))

        # Projection matrix
        self.mat_proj = self.build_projection(self.cam, 0, 0, size[0], size[1], clip_near, clip_far)

        # Set up shader programs
        self.type = type
        if self.type == 'rgbd' or self.type == 'rgbc' or self.type == 'corr':
            self.program = gloo.Program(_vertex_code, _fragment_code_rgb)
        elif self.type == 'normals':
            self.program = gloo.Program(_vertex_code, _fragment_code_normals)

        gloo.set_state(preset='translucent')
        gloo.set_clear_color((0.0, 0.0, 0.0, 0.0))
        gloo.set_viewport(0, 0, *self.size)

        # Smooth
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glEnable(gl.GL_POLYGON_SMOOTH)
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
        gl.glHint(gl.GL_POLYGON_SMOOTH_HINT, gl.GL_NICEST)

        # Multisampling
        gl.glEnable(gl.GL_MULTISAMPLE)


    def clear(self):
        gloo.clear(color=True, depth=True)

    def draw_background(self, color=None, depth=None):
        pass  # works only if nothing is drawn anymore, otherwise deleted?!?!

        gloo.set_state(depth_test=False)
        filling = np.zeros(self.render_tex.shape, dtype=np.float32)
        if color is not None:
            filling[:, :, 0:3] = color[::-1, :]
        if depth is not None:
            filling[:, :, 3] = depth[::-1]
        gl.glDrawPixels(self.size[0], self.size[1], gl.GL_RGBA, gl.GL_FLOAT, filling)
        gloo.set_state(depth_test=True)

    def finish(self):
        im = gl.glReadPixels(0, 0, self.size[0], self.size[1], gl.GL_RGBA, gl.GL_FLOAT)
        rgb = np.copy(np.frombuffer(im, np.float32)).reshape(self.shape + (4,))[::-1, :]  # Read buffer and flip Y
        im = gl.glReadPixels(0, 0, self.size[0], self.size[1], gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)
        dep = np.copy(np.frombuffer(im, np.float32)).reshape(self.shape + (1,))[::-1, :]  # Read buffer and flip Y
        # Convert z-buffer to depth map
        mult = (self.clip_near * self.clip_far) / (self.clip_near - self.clip_far)
        addi = self.clip_far / (self.clip_near - self.clip_far);
        bg = dep == 1
        dep = mult / (dep + addi)
        dep[bg] = 0
        return rgb, dep

    def draw_model(self, model, pose, ambient_weight=1, light=(0, 0, 0), output='triangles'):

        # Light augmentation
        light = list(np.random.uniform(-200, 200, (3,)))
        color = list(np.random.uniform(.85, 1.0, (3,)))

        # View matrix (transforming the coordinate system from OpenCV to OpenGL camera space)
        mv = (self.yz_flip.dot(pose)).T  # OpenCV to OpenGL camera system (flipped, column-wise)
        mvp = mv.dot(self.mat_proj)

        self.program.bind(model.vertex_buffer)
        self.program['u_light_eye_pos'] = light
        self.program['u_mv'] = mv
        self.program['u_mvp'] = mvp
        self.program.draw(output, model.index_buffer)

    def build_projection(self, cam, x0, y0, w, h, nc, fc):

        q = -(fc + nc) / float(fc - nc)
        qn = -2 * (fc * nc) / float(fc - nc)

        # Draw our images upside down, so that all the pixel-based coordinate systems are the same
        proj = np.array([
            [2 * cam[0, 0] / w, -2 * cam[0, 1] / w, (-2 * cam[0, 2] + w + 2 * x0) / w, 0],
            [0, -2 * cam[1, 1] / h, (-2 * cam[1, 2] + h + 2 * y0) / h, 0],
            [0, 0, q, qn],  # This row is standard glPerspective and sets near and far planes
            [0, 0, -1, 0]
        ])

        # Compensate for the flipped image
        proj[1, :] *= -1.0
        return proj.T

    def render_views(self, model, poses, output='triangles'):
        color = []
        depth = []

        for pose in poses:
            self.clear()
            self.draw_model(model, pose, output=output)
            col, dep = self.finish()
            color.append(col)
            depth.append(dep)

        return color, depth

    def render_views_store(self, model, poses, folder='images'):

        for pose_id, pose in enumerate(poses):
            self.clear()
            self.draw_model(model, pose)
            col, dep = self.finish()

            util.store_images(col, dep, pose, pose_id, type=self.type, folder=folder)
