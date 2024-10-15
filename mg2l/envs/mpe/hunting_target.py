import numpy as np
from gym.spaces import Box


class HuntingTargetMPE:
    def __init__(self, n_agents, n_tasks, max_episode_steps=25, task=None, **kwargs):
        from .multiagent.environment import MultiAgentEnv
        from .multiagent.scenarios.hunting import Scenario
        scenario = Scenario()
        world = scenario.make_world(n_agents, n_tasks)
        self.env = MultiAgentEnv(world=world,
                                 reset_callback=scenario.reset_world,
                                 reward_callback=scenario.reward,
                                 observation_callback=scenario.observation, )
        self.n_agents = n_agents

        self.action_space = self.env.action_space[:self.n_agents]
        self.observation_space = self.env.observation_space[:self.n_agents]
        share_obs_dim = sum([self.env.observation_space[i].shape[0] for i in range(self.n_agents)])
        self.share_observation_space = [Box(low=np.array([-np.inf] * share_obs_dim, dtype=np.float32),
                                            high=np.array([np.inf] * share_obs_dim, dtype=np.float32),
                                            dtype=np.float32) for _ in range(self.n_agents)]

        self.tasks = [{"effective": i} for i in range(n_tasks)]
        self._task = self.tasks[0] if task is None else task
        self.num_tasks = n_tasks
        self.reset_task(0)

    def step(self, actions):
        action_preys = []
        for agent in self.env.agents:
            if "prey" in agent.name:
                action_preys.append(self.prey_action(agent))
        action_env = list(actions) + action_preys
        obs_n, reward_n, done_n, info = self.env.step(action_env)
        return obs_n[:self.n_agents], reward_n[:self.n_agents], done_n[:self.n_agents], info

    def reset_task(self, idx=None):
        if idx is None:
            idx = np.random.randint(self.num_tasks)
        self._task = self.tasks[idx]
        for agent in self.env.agents:
            agent.target = True if agent.name == 'prey %d' % self._task["effective"] else False

    def reset(self):
        return self.env.reset()[:self.n_agents]

    def render(self, mode="human"):
        return self.env.render(mode)

    def prey_action(self, prey):
        action = np.random.rand(self.env.world.dim_p)
        if not prey.target:
            return action

        min_dist = np.inf
        for agent in self.env.agents:
            if "predator" in agent.name:
                delta_pos = prey.state.p_pos - agent.state.p_pos
                dist = np.linalg.norm(delta_pos)
                if dist < min_dist:
                    min_dist = dist
                    action = delta_pos / dist
        return action

    def close(self):
        self.env.close()

