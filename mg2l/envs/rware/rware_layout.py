import numpy as np
import gym
from gym.spaces import Box
import rware


class MultiMapRware:
    def __init__(self, n_agents, n_tasks=4, max_episode_steps=25, layouts=None, task=None, **kwargs):
        assert len(layouts) >= n_tasks
        if "difficulty" not in kwargs:
            self.difficulty = ""
        else:
            self.difficulty = kwargs["difficulty"]
        if "sensor_range" not in kwargs:
            self.sensor_range = 1
        else:
            self.sensor_range = kwargs["sensor_range"]

        self.env_id = "rware-tiny-%dag%s-v1" % (n_agents, self.difficulty)
        self.mm_layouts = layouts[:n_tasks]
        self.mm_envs = [gym.make(self.env_id,
                                 layout=self.mm_layouts[i],
                                 sensor_range=self.sensor_range
                                 )
                        for i in range(n_tasks)]
        self.n_agents = n_agents

        self.action_space = self.mm_envs[0].action_space
        self.observation_space = self.mm_envs[0].observation_space
        share_obs_dim = sum([self.observation_space[i].shape[0] for i in range(n_agents)])
        self.share_observation_space = [Box(low=np.array([-np.inf] * share_obs_dim, dtype=np.float32),
                                            high=np.array([np.inf] * share_obs_dim, dtype=np.float32),
                                            dtype=np.float32) for _ in range(n_agents)]

        self.tasks = [{"layout": i} for i in range(n_tasks)]
        self._task = self.tasks[0] if task is None else task
        self.num_tasks = n_tasks
        self.reset_task(0)

    def step(self, actions):
        obs_n, reward_n, done_n, info = self.mm_envs[self._task["layout"]].step(actions)
        return obs_n, reward_n, done_n, info

    def reset_task(self, idx=None):
        if idx is None:
            idx = np.random.randint(self.num_tasks)
        self._task = self.tasks[idx]

    def reset(self):
        return self.mm_envs[self._task["layout"]].reset()

    def render(self, mode="human"):
        return self.mm_envs[self._task["layout"]].render(mode)

    def close(self):
        return self.mm_envs[self._task["layout"]].close()

