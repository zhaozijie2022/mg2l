import os
import sys
import torch
from copy import deepcopy
import numpy as np
from omegaconf import OmegaConf

import setproctitle

import mg2l.utils.pytorch_utils as ptu
from learner import Learner
from utils.config_utils import get_config, with_cover, sort_cfg, print_cfg


if __name__ == "__main__":
    params = deepcopy(sys.argv)


    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
    expt_cfg = get_config(params, "--expt", config_path, "expt_config")
    env_cfg = get_config(params, "--env", config_path, "env_config")
    algo_cfg = get_config(params, "--algo", config_path, "algo_config")
    # region parse args

    cfg = OmegaConf.merge(env_cfg, algo_cfg, expt_cfg)
    cfg = with_cover(params, cfg)
    cfg = sort_cfg(cfg)
    print_cfg(cfg)

    ptu.set_gpu_mode(False, gpu_id=cfg.gpu_id)
    print("Cuda is available: {}, gpu_id: {}".format(ptu.gpu_enabled(), ptu.gpu_id()))
    torch.set_num_threads(cfg.n_training_threads)
    setproctitle.setproctitle("mg2l-{}-{}".format(cfg.save_name, cfg.algo_name))
    os.makedirs(cfg.main_save_path, exist_ok=True)

    learner = Learner(cfg)
    learner.train()


