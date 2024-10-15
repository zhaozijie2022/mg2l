import os
from omegaconf import OmegaConf, DictConfig


def get_config(params, arg_name, config_path, sub_folder) -> DictConfig:
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        cfg = OmegaConf.load(os.path.join(config_path, sub_folder, "{}.yaml".format(config_name)))
        if "multitask_mujoco" in config_name:
            cfg = supply_env_args(cfg)
        return cfg



def supply_env_args(env_cfg: DictConfig) -> DictConfig:
    task_idxes = env_cfg.which_task.split("_")
    task_idxes = [int(idx) for idx in task_idxes]

    name_list = []
    num_agents_list = []
    for i in range(len(task_idxes)):
        name_list.append(env_cfg.env_names[task_idxes[i]])
        name_list[-1]["episode_limit"] = env_cfg.max_ep_len
        num_agents_list.append(int(name_list[-1]["env_name"].split('_')[0]))

    env_cfg.env_names = name_list
    env_cfg.n_tasks = len(task_idxes)
    env_cfg.n_agents = max(num_agents_list)
    return env_cfg


def with_cover(params, config_dict: DictConfig) -> DictConfig:
    for _i, _v in enumerate(params):
        if "=" in _v:
            _key = _v.split("=")[0]
            _value = _v.split("=")[1]

            if _value.lower() == "true":
                _value = True
            elif _value.lower() == "false":
                _value = False
            elif _value.isdigit():
                _value = int(_value)

            if _key in config_dict:
                config_dict[_key] = _value
            else:
                config_dict[_key] = _value
                print("Warning: {} is not in config".format(_key))

    return config_dict


def sort_cfg(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create({k: cfg[k] for k in sorted(cfg.keys())})


def print_cfg(cfg: DictConfig):
    _cfg = OmegaConf.to_container(cfg)
    print('=' * 20 + "Config:" + '=' * 20)
    for k, v in _cfg.items():
        print(k, ':', v)
    print("")

