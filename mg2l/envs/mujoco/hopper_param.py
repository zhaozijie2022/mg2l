import numpy as np
from .multiagent_mujoco.mujoco_multi import MujocoMulti


class HopperParamEnvMulti(MujocoMulti):
    def __init__(self, n_agents=3, n_tasks=10, max_episode_steps=1000, task=None, **kwargs):

        if "agent_obsk" not in kwargs:
            kwargs["agent_obsk"] = 0
        if n_agents == 3:
            env_args = {"scenario": "Hopper-v2",
                        "agent_conf": "3x1",
                        "agent_obsk": kwargs["agent_obsk"],
                        "episode_limit": max_episode_steps}
        else:
            raise NotImplementedError("num of agents: %d doesn't match Hopper" % n_agents)

        self.tasks = [{"param": np.random.uniform(low=0.5, high=1.5, size=(4,))} for _ in range(n_tasks)]
        self._task = self.tasks[0] if task is None else task
        self.num_tasks = n_tasks

        super().__init__(env_args=env_args)
        self.reset_task(0)

    def step(self, actions):
        reward, done, info = super().step(actions)
        obs_n = self.get_obs()

        reward_n = [np.array(reward) for _ in range(self.n_agents)]
        done_n = [done for _ in range(self.n_agents)]
        return obs_n, reward_n, done_n, info

    def reset_task(self, idx=None):
        if idx is None:
            idx = np.random.randint(self.num_tasks)
        self._task = self.tasks[idx]

        bm = self.env.model.body_mass
        bm *= self._task["param"][0]

        bd = self.env.model.dof_damping
        bd *= self._task["param"][1]

        bf = self.env.model.geom_friction
        bf *= self._task["param"][2]

        bi = self.env.model.body_inertia
        bi *= self._task["param"][3]



