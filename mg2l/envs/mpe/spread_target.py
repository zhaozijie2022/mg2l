import numpy as np
from gym.spaces import Box


class SpreadTargetMPE:
    def __init__(self, n_agents, n_tasks, max_episode_steps=25, task=None, **kwargs):
        from .multiagent.environment import MultiAgentEnv
        from .multiagent.scenarios.spread import Scenario
        scenario = Scenario()
        world = scenario.make_world(n_agents, n_tasks)
        self.env = MultiAgentEnv(world=world,
                                 reset_callback=scenario.reset_world,
                                 reward_callback=scenario.reward,
                                 observation_callback=scenario.observation, )
        self.n_agents = n_agents

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        share_obs_dim = sum([self.env.observation_space[i].shape[0] for i in range(self.n_agents)])
        self.share_observation_space = [Box(low=np.array([-np.inf] * share_obs_dim, dtype=np.float32),
                                            high=np.array([np.inf] * share_obs_dim, dtype=np.float32),
                                            dtype=np.float32) for _ in range(self.n_agents)]

        self.tasks = self.sample_tasks(n_tasks)
        self._task = self.tasks[0] if task is None else task
        self.num_tasks = n_tasks
        self.reset_task(0)

    def step(self, actions):
        obs_n, reward_n, done_n, info = self.env.step(list(actions))
        return obs_n, reward_n, done_n, info

    def sample_tasks(self, n_tasks):
        import itertools

        n_landmarks = self.n_agents
        while len(list(itertools.combinations(range(n_landmarks), self.n_agents))) < n_tasks:
            n_landmarks += 1

        all_choices = list(itertools.combinations(range(n_landmarks), self.n_agents))
        np.random.shuffle(all_choices)
        return [{"effective": np.array([1 if i in all_choices[j] else 0 for i in range(n_landmarks)])} for j in range(n_tasks)]

    def reset_task(self, idx=None):
        if idx is None:
            idx = np.random.randint(self.num_tasks)
        self._task = self.tasks[idx]
        for landmark in self.env.world.landmarks:
            l_id = int(landmark.name[-1])
            if self._task["effective"][l_id] == 1:
                landmark.target = True
            else:
                landmark.target = False

    def reset(self):
        return self.env.reset()

    def render(self, mode="human"):
        return self.env.render(mode)

