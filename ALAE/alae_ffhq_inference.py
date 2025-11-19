from ALAE.model import Model
from ALAE.checkpointer import Checkpointer
from ALAE.defaults import get_cfg_defaults
import ALAE.lreq as lreq
import numpy as np
import argparse
import logging
import torch
import os
from tqdm import tqdm
from matplotlib import pyplot as plt

def load_model(default_config, training_artifacts_dir):
    lreq.use_implicit_lreq.set(True)

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-c", "--config-file",
        default=default_config,
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args(args=["OUTPUT_DIR", training_artifacts_dir])
    cfg = get_cfg_defaults()

    config_file = args.config_file
    if len(os.path.splitext(config_file)[1]) == 0:
        config_file += '.yaml'
    if not os.path.exists(config_file) and os.path.exists(os.path.join('configs', config_file)):
        config_file = os.path.join('configs', config_file)

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
        truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER
    )
    model.eval()
    model.requires_grad_(False)

    model_dict = {
        'discriminator_s': model.encoder,
        'generator_s': model.decoder,
        'mapping_tl_s': model.mapping_d,
        'mapping_fl_s': model.mapping_f,
        'dlatent_avg': model.dlatent_avg
    }

    logger = logging.getLogger("logger")
    checkpointer = Checkpointer(cfg, model_dict, {}, logger=logger, save=False)
    _ = checkpointer.load()

    return model

def encode(model, x):
    layer_count = model.layer_count
    Z, _ = model.encode(x, layer_count - 1, 1)
    Z = Z.repeat(1, model.mapping_f.num_layers, 1)
    return Z

def decode(model, x):
    x = x[:, None, :].repeat(1, model.mapping_f.num_layers, 1)
    layer_count = model.layer_count
    decoded = model.decoder(x, layer_count - 1, 1, noise=True)
    return decoded
