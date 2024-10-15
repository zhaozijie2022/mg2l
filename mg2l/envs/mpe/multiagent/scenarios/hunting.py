import numpy as np
from ..core import World, Agent, Landmark
from ..scenario import BaseScenario



class Scenario:
    def make_world(self, n_predators, n_preys):
        world = World()
        world.bb = 1.2
        world.boundary = [np.array([world.bb, 0]), np.array([-world.bb, 0]),
                          np.array([0, world.bb]), np.array([0, -world.bb])]
        world.wall = [np.array([world.bb, world.bb]), np.array([-world.bb, world.bb]),
                      np.array([-world.bb, -world.bb]), np.array([world.bb, -world.bb])]
        # set any world properties first
        world.dim_c = 2
        world.target_id = 0
        # add agents
        predators = [Agent() for _ in range(n_predators)]
        world.n_predators = n_predators
        for i, predator in enumerate(predators):
            predator.name = 'predator %d' % i
            predator.size = 0.1
            predator.adversary = True
            predator.target = False
            predator.accel = 3.0
            predator.max_speed = 1.0

        preys = [Agent() for _ in range(n_preys)]
        world.n_preys = n_preys
        for i, prey in enumerate(preys):
            prey.name = 'prey %d' % i
            prey.size = 0.05
            prey.adversary = False
            prey.target = True if i == world.target_id else False
            prey.accel = 3.0
            prey.max_speed = 1.2

        world.agents = predators + preys
        for i, agent in enumerate(world.agents):
            agent.collide = True
            agent.silent = True

        world.landmarks = []
        self.reset_world(world)
        return world

    def reset_world(self, world):
        for i, agent in enumerate(world.agents):
            if agent.target:
                agent.color = np.array([0.85, 0.35, 0.35])
            elif agent.adversary:
                agent.color = np.array([0.35, 0.85, 0.35])
            else:
                agent.color = np.array([0.35, 0.35, 0.85])
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        for i, agent in enumerate(world.agents):
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def preys(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    def predators(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        vertices = [pred.state.p_pos for pred in self.predators(world)]
        target = [prey for prey in self.preys(world) if prey.target][0]
        point = target.state.p_pos
        reward = 0.
        reward -= 0.1 * np.sum(np.linalg.norm(point - vertices, axis=1))
        reward += 10 * self.is_collision(agent, target)
        return reward

    def observation(self, agent, world):
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        other_pos = []
        other_dist = []
        for other in world.agents:
            if other is agent:
                continue
            _dist = np.linalg.norm(other.state.p_pos - agent.state.p_pos)
            if _dist < 1.414:
                other_pos.append(other.state.p_pos - agent.state.p_pos)
                other_dist.append([_dist])
            else:
                other_pos.append(np.zeros(world.dim_p))
                other_dist.append([0.])

        return np.concatenate([agent.state.p_pos] + entity_pos + other_pos + other_dist)
