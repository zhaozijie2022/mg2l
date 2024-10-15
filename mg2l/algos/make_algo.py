import importlib
import mg2l.utils.pytorch_utils as ptu
from mg2l.algos.mappo import MAPPOTrainer, MAPPOPolicy


def make_algo(cfg):
    policy = []
    for agent_id in range(cfg.n_agents):
        share_observation_space = (
            cfg.share_observation_space[agent_id]
            if cfg.use_centralized_V
            else cfg.observation_space[agent_id]
        )
        # policy network
        po = MAPPOPolicy(
            cfg,
            cfg.observation_space[agent_id],
            share_observation_space,
            cfg.action_space[agent_id],
            agent_id,
        )
        policy.append(po)
    return [MAPPOTrainer(
        cfg=cfg,
        policy=policy[i],
        agent_id=i,
    ) for i in range(cfg.n_agents)]









