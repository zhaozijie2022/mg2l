import numpy as np
from ..core import World, Agent, Landmark
from ..scenario import BaseScenario


class Scenario:
    def make_world(self, n_agents=3, n_tasks=4):
        assert n_tasks > n_agents

        world = World()
        world.bb = 1.2
        world.boundary = [np.array([world.bb, 0]), np.array([-world.bb, 0]),
                          np.array([0, world.bb]), np.array([0, -world.bb])]
        world.wall = [np.array([world.bb, world.bb]), np.array([-world.bb, world.bb]),
                      np.array([-world.bb, -world.bb]), np.array([world.bb, -world.bb])]
        # set any world properties first
        world.dim_c = 2
        world.collaborative = True
        # add agents
        world.agents = [Agent() for _ in range(n_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.size = 0.15

        world.landmarks = [Landmark() for _ in range(n_tasks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.target = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            if landmark.target:
                landmark.color = np.array([0.85, 0.25, 0.25])
            else:
                landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        rew = 0.
        for l in world.landmarks:
            if l.target:
                dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
                rew -= min(dists)
        return rew

    def observation(self, agent, world):
        entity_pos = []
        entity_dist = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            entity_dist.append(np.linalg.norm(entity.state.p_pos - agent.state.p_pos))

        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_pos] + entity_pos + other_pos)

