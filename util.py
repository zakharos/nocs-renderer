import numpy as np
import cv2
import os
import util


def read_cfg_string(cfg, section, key, default):
    if cfg.has_option(section, key):
        return cfg.get(section, key)
    else:
        return default


def read_cfg_int(cfg, section, key, default):
    if cfg.has_option(section, key):
        return cfg.getint(section, key)
    else:
        return default


def read_cfg_float(cfg, section, key, default):
    if cfg.has_option(section, key):
        return cfg.getfloat(section, key)
    else:
        return default


def read_cfg_bool(cfg, section, key, default):
    if cfg.has_option(section, key):
        return cfg.get(section, key) in ['True', 'true']
    else:
        return default


def read_cfg_cam(cfg, section, key, default):
    if cfg.has_option(section, key):
        str = cfg.get(section, key).split(',')
        cam = np.array([[float(str[0]), 0., float(str[1])],
                        [0., float(str[2]), float(str[3])],
                        [0., 0., 1.]])
        return cam
    else:
        return default


def read_cfg_tuple(cfg, section, key, default):
    if cfg.has_option(section, key):
        str = cfg.get(section, key).split(',')
        tup = (int(str[0]), int(str[1]))
        return tup
    else:
        return default


def normalize_depth(patch, clamp, z):
    patch = np.clip(patch - z, -clamp, clamp)  # Demean and clamp
    return ((patch / clamp) * 0.5) + 0.5  # Normalize to [0,1]


def get_tight_bbox(dep, delta=5):
    a = np.where(dep != 0)
    bbox = []
    bbox.append(max(0, np.min(a[0]) - delta))  # bottom
    bbox.append(max(0, np.min(a[1]) - delta))  # left
    bbox.append(min(np.max(a[0]) + delta, dep.shape[0]))  # top
    bbox.append(min(np.max(a[1]) + delta, dep.shape[1]))  # right

    return bbox


def normalize(x):
    return (x - min(x)) / (max(x) - min(x))


def store_images(color, depth, pose, pose_id, type='rgbd', folder='images', prefix='synth'):
    if not os.path.exists(folder):
        os.makedirs(folder)

    cut_patch = util.get_tight_bbox(depth)
    center = np.zeros_like(depth)
    center[int(color.shape[0]/2), int(color.shape[1]/2)] = 1
    center = center[cut_patch[0]:cut_patch[2], cut_patch[1]:cut_patch[3]]
    color = color[cut_patch[0]:cut_patch[2], cut_patch[1]:cut_patch[3]]
    depth = depth[cut_patch[0]:cut_patch[2], cut_patch[1]:cut_patch[3]]

    print("Saving view #{}".format(pose_id))
    img_path = os.path.join(folder, prefix + "_" + str(format(pose_id, "04")))

    if type == 'rgbc':
        cv2.imwrite(img_path + "_img.png", color * 255)
        cv2.imwrite(img_path + "_center.png", (center * 255))

    if type == 'corr':
        cv2.imwrite(img_path + "_corr.png", color[:, :, :3] * 255)

    elif type == 'rgbd':
        # Normalize depth
        depth = normalize_depth(depth, 0.3, pose[2, 3])

        cv2.imwrite(img_path + "_img.png", color * 255)
        cv2.imwrite(img_path + "_dpt_vis.png", (depth * 255))
        cv2.imwrite(img_path + "_dpt.png", (depth * 1000).astype(np.uint16))

    elif type == 'normals':
        cv2.imwrite(img_path + "_nor.png", color[:, :, :3] * 255)
