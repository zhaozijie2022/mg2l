import numpy as np

from .multiagent_mujoco.mujoco_multi import MujocoMulti


class HalfCheetahDirEnvMulti(MujocoMulti):
    def __init__(self, n_agents, n_tasks=2, max_episode_steps=1000, task=None, **kwargs):
        if "agent_obsk" not in kwargs:
            kwargs["agent_obsk"] = 0
        if n_agents == 2:
            env_args = {"scenario": "HalfCheetah-v2",
                        "agent_conf": "2x3",
                        "agent_obsk": kwargs["agent_obsk"],
                        "episode_limit": max_episode_steps}
        elif n_agents == 6:
            env_args = {"scenario": "HalfCheetah-v2",
                        "agent_conf": "6x1",
                        "agent_obsk": kwargs["agent_obsk"],
                        "episode_limit": max_episode_steps}
        else:
            raise NotImplementedError("num of agents: %d doesn't match HalfCheetah" % n_agents)

        self._task = {"goal": 1.} if task is None else task
        self.num_tasks = n_tasks
        self.tasks = [{"goal": -1.0}, {"goal": 1.0}]

        super().__init__(env_args=env_args)
        self.reset_task(0)

    def step(self, actions):
        reward, done, info = super().step(actions)
        obs_n = self.get_obs()

        reward = self._task["goal"] * info['reward_run'] + info['reward_ctrl']
        reward_n = [np.array(reward) for _ in range(self.n_agents)]
        done_n = [done for _ in range(self.n_agents)]
        return obs_n, reward_n, done_n, info

    def reset_task(self, task_idx=None):
        if task_idx is None:
            task_idx = np.random.randint(self.num_tasks)
        self._task = self.tasks[task_idx]









