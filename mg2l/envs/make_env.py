import os

import gym
import numpy as np
import torch
from gym.spaces.box import Box
import importlib
from .wrappers import DummyVecEnv, SubprocVecEnv


def make_env(cfg):
    def get_env_fn(rank):
        def init_env():
            if "mujoco" in cfg.env_file:
                env_file = importlib.import_module("mg2l.envs." + cfg.env_file)
                Env = getattr(env_file, cfg.env_class)
                env = Env(n_agents=cfg.n_agents,
                          n_tasks=cfg.n_tasks,
                          max_episode_steps=cfg.max_ep_len,
                          agent_obsk=cfg.agent_obsk, )

                if cfg.seed is not None:
                    env.env.seed(cfg.seed + 1024 * rank)
                return env
            elif "mpe" in cfg.env_file:
                import warnings
                warnings.filterwarnings("ignore", category=DeprecationWarning)

                env_file = importlib.import_module("mg2l.envs." + cfg.env_file)
                Env = getattr(env_file, cfg.env_class)
                env = Env(n_agents=cfg.n_agents,
                          n_tasks=cfg.n_tasks,
                          max_episode_steps=cfg.max_ep_len, )
                return env
            elif "rware" in cfg.env_file:
                env_file = importlib.import_module("mg2l.envs." + cfg.env_file)
                Env = getattr(env_file, cfg.env_class)
                env = Env(n_agents=cfg.n_agents,
                          n_tasks=cfg.n_tasks,
                          max_episode_steps=cfg.max_ep_len,
                          layouts=cfg.layouts,
                          difficulty=cfg.difficulty,
                          sensor_range=cfg.sensor_range,)
                return env
            else:
                raise NotImplementedError("env_file: %s not found" % cfg.env_file)

        return init_env

    if cfg.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(cfg.n_rollout_threads)])
