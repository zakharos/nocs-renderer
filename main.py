import os
import sys
import configparser
import argparse
import sampling
from datatype import Model
from renderer import Renderer
import util


def main():

    # set up a parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=False, default="config.ini", help="Config file")

    # parse arguments
    args = parser.parse_args()
    cfg = configparser.ConfigParser()
    res = cfg.read(args.config)
    if len(res) == 0:
        print("Error: None of the config files could be read")
        sys.exit(1)

    # read the config
    model_path = util.read_cfg_string(cfg, 'input_output', 'model_path', default=None)
    model_scale = util.read_cfg_string(cfg, 'input_output', 'model_scale', default='mm')
    output_types = util.read_cfg_string(cfg, 'renderer', 'target_types', default='rgbd,normals,nocs').split(',')
    output_folder = util.read_cfg_string(cfg, 'input_output', 'output_folder', default='output')
    os.makedirs(output_folder, exist_ok=True)

    # initialize sampling, renderer and noise objects
    icosahedron = sampling.Icosahedron(cfg)
    poses = icosahedron.create_poses()

    for output_type in output_types:
        renderer = Renderer(cfg, output_type)
        print("Rendering {}, model: {}".format(output_type, model_path))

        # load a model
        model = Model()
        model.load(model_path, color=output_type, scale=model_scale)

        # render and store a model
        renderer.render_views_store(model, poses, folder=output_folder)


if __name__ == "__main__":
    main()
