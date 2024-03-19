import functools
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random
from Agent import Agent
import time
import pyglet
from gridworld import GridMatrix, Grid
from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, MultiDiscrete, Dict, Box
from copy import copy

class MAGridWorldEnv(ParallelEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, n_width: int = 10,
                 n_height: int = 7,
                 u_size=40,
                 default_reward: float = 0,
                 default_type=0,
                 n_agents=2,
                 n_targets=3):

        self.u_size = u_size  # size for each cell (pixels)
        self.n_width = n_width  # width of the env calculated by number of cells.
        self.n_height = n_height  # height...
        self.width = u_size * n_width  # scenario width (pixels)
        self.height = u_size * n_height  # height
        self.default_reward = default_reward
        self.default_type = default_type

        self.grids = GridMatrix(n_width=self.n_width,
                                n_height=self.n_height,
                                default_reward=self.default_reward,
                                default_type=self.default_type,
                                default_value=self.default_type)

        self.n_agents = n_agents  # 智能体数量
        self.n_targets = n_targets  # 目标数量
        self.targets = [[3, 2], [5, 1], [7, 4]]  # 目标位置
        self.start = {f"agent_{i}": 0 for i in range(n_agents)}
        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)] #并不是很明白这一行和下面一行有什么区别
        self.agents = [f"agent_{i}" for i in range(self.n_agents)]
        # 创建多个智能体实例
        # self.agents_list = {f"agent_{i}":Agent(self.start[agent_id], agent_id) for i, agent_id in enumerate(self.possible_agents)}
        # self.pos = [0] * n_agents  # 用于存储每个智能体的状态
        self.pos = {f"agent_{i}": 0 for i in range(n_agents)}
        self.tpm = [[0] * (self.n_width * self.n_height) for i in range(n_agents)]  # 用于存储每个智能体的观察,一维或二维暂时存疑
        self.direct = {f"agent_{i}": [0, 1] for i in range(n_agents)}  # 用于存储每个智能体的朝向
        self._pos = {f"agent_{i}": [0] * (n_agents - 1) for i in range(n_agents)}
        self._direct = {f"agent_{i}": [0, 1] * (n_agents - 1) for i in range(n_agents)}
        # self.start = [0, 0]*n_agents
        # print(self.pos)

        self.action = {f"agent_{i}": 0 for i in range(n_agents)}
        self.reward = {f"agent_{i}": 0 for i in range(n_agents)}
        # self.directions = [[0, 1]]*n_agents
        # 方向向量，用于计算智能体的移动。默认朝上
        self.set_start()
        # self.set_
        self.viewer = None  # 图形接口对象
        self.agent_geoms = []  # Initialize the list for agent rendering objects
        self.seed()  # 产生一个随机子
        self.timestep = 0  # 记录当前的时间步
        self.reset()

    def seed(self, seed=None):
        # 产生一个随机化时需要的种子，同时返回一个np_random对象，支持后续的随机化生成操作
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_start(self):
        # 清空start列表
        for i, agent in enumerate(self.agents):
            # 随机设置每个智能体的起始位置
            # self.start.append((random.randint(0, self.n_width - 1), random.randint(0, self.n_height - 1)))
            self.start[agent] = self._xy_to_state(random.randint(0, self.n_width - 1), random.randint(0, self.n_height - 1))

    def set_target(self, target_list):
        for i in range(self.n_targets):
            self.grids.set_type(target_list[i][0], target_list[i][1], 2)
            self.grids.set_reward(target_list[i][0], target_list[i][1], 10)

    def reset(self,seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0
        for i, agent in enumerate(self.agents):
            # agent.position = self.start[agent]  # 重置所有智能体的位置
            self.pos[agent] = self.start[agent]  # 设置当前智能体的状态
            # agent.direct = [0, 1]  # 设置朝向
            self.direct[agent] = [0, 1]  # 设置朝向
            # next_state[i].append(self.tpm)
            # next_state[i].append(self.pos[i])
            # next_state[i].append(agent.direct)

            for i,agent in enumerate(self._pos):
                self._pos[agent] = [0] * (self.n_agents - 1)
                self._direct[agent] = [0, 1] * (self.n_agents - 1)
            # for j, _agent in enumerate(self.agents):
            #     if agent != _agent: _pos.append(self.pos[_agent])

            # for j, agent in enumerate(self.agents):
            #     if i != j: _state.append(self.agents[j].direct)
        observation = {
            a: (
                self.tpm,
                self.pos[a],
                self._pos[a],
                self.direct[a],
                self._direct[a]
            )
            for a in self.agents
        }

        infos = {a: {} for a in self.agents}

        return observation,infos
        # 可以添加更多的重置逻辑

    def step(self, actions):
        rewards = {a: 0 for a in self.agents}
        # dones = {a: False for a in self.agents}
        pos = {a: 0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}

        FORWARD = 0
        LEFT_FORWARD = 1
        RIGHT_FORWARD = 2

        # 定义朝向变化
        direction_changes = {
            LEFT_FORWARD: [-1, 1],  # 向左转90度
            RIGHT_FORWARD: [1, -1]  # 向右转90度
        }

        def go_left(direction):
            if direction == [0, 1]:
                return [-1, 1]
            elif direction == [-1, 0]:
                return [-1, -1]
            elif direction == [0, -1]:
                return [1, -1]
            elif direction == [1, 0]:
                return [1, 1]

        def go_right(direction):
            if direction == [0, 1]:
                return [1, 1]
            elif direction == [1, 0]:
                return [1, -1]
            elif direction == [0, -1]:
                return [-1, -1]
            elif direction == [-1, 0]:
                return [-1, 1]

        # for i, action in enumerate(actions):
        for agent, act in actions.items():
            # time.sleep(0.5)
            self.action[agent] = actions[agent]  # action for rendering
            old_x, old_y = self._state_to_xy(self.pos[agent])
            new_x, new_y = old_x, old_y

            if act in direction_changes:
                if act == LEFT_FORWARD:
                    print(agent + " is turning left")
                elif act == RIGHT_FORWARD:
                    print(agent + " is turning right")
                # 计算新朝向
                change = direction_changes[act]
                # self.agents[i].direct = [self.agents[i].direct[1] * change[0], self.agents[i].direct[0] * change[1]]
                self.direct[agent] = [self.direct[agent][1] * change[0], self.direct[agent][0] * change[1]]
            # 沿当前朝向前进一格
            # new_x, new_y = new_x + self.agents[i].direct[0], new_y + self.agents[i].direct[1]
            new_x, new_y = new_x + self.direct[agent][0], new_y + self.direct[agent][1]
            # boundary effect
            if new_x < 0: truncations[agent] = True
            if new_x >= self.n_width: truncations[agent] = True
            if new_y < 0: truncations[agent] = True
            if new_y >= self.n_height: truncations[agent] = True

            pos[agent] = self._xy_to_state(new_x, new_y)
            # wall effect, obstacles or boundary.
            # 类型为1的格子为障碍格子，不可进入
            if self.grids.is_existed(new_x, new_y)== False or self.grids.get_type(new_x, new_y) == 1:
                # new_x, new_y = old_x, old_y
                truncations[agent] = True

            tpm = []
            # 处理智能体周围一圈格子
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if self.grids.is_existed(new_x + x, new_y + y):
                        # and self.grids.get_type(new_x + x, new_y + y) == 0):
                        self.grids.set_value(new_x + x, new_y + y,
                                             1)  # 设置为已访问
                        rewards[agent] += self.grids.get_reward(new_x + x, new_y + y)  # 获取奖励
                        self.grids.set_reward(new_x + x, new_y + y, 0)  # 获取奖励后将奖励置0
            for x in range(self.n_width):
                for y in range(self.n_height):
                    tpm.append(self.grids.get_value(x, y))  # 获取观察

            # self.pos[agent] = self._xy_to_state(new_x, new_y)
            self.pos[agent] = pos[agent]
            self.tpm = tpm
            self._pos[agent] = [self.pos[_agent] for _agent in self.agents if _agent != agent]
            # self._direct[agent] = [self.direct[_agent] for _agent in self.agents if _agent != agent]
            self._direct[agent] = []
            for _agent in self.agents:
                if _agent != agent:
                    self._direct[agent].extend(self.direct[_agent])
            self.reward[agent] = rewards[agent]

        obs = {
            a: (
                self.tpm,
                self.pos[a],
                self._pos[a],
                self.direct[a],
                self._direct[a]
            )
            for a in self.agents
        }
        print("all stepped!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        infos = {a: {} for a in self.agents}
        self.timestep += 1

        # 查看当前episode是否应该结束
        if any(terminations.values()) or any(truncations.values()):
            if any(truncations.values()):
                print("Truncated")
                # 设置truncations全为true
                truncations = {a: True for a in self.agents}
            elif any(terminations.values()):
                print("Terminated")
                # 设置terminations全为true
                terminations = {a: True for a in self.agents}
            self.agents = []

        return obs, self.reward, terminations, truncations, infos

    from gym.envs.classic_control import rendering

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.width, self.height)

            # 初始化格子
            self.grid_geoms = []
            for x in range(self.n_width):
                for y in range(self.n_height):
                    rect = rendering.FilledPolygon([(x * self.u_size, y * self.u_size),
                                                    ((x + 1) * self.u_size, y * self.u_size),
                                                    ((x + 1) * self.u_size, (y + 1) * self.u_size),
                                                    (x * self.u_size, (y + 1) * self.u_size)])
                    self.viewer.add_geom(rect)
                    self.grid_geoms.append(rect)

            # 绘制格子间的分隔线
            for x in range(self.n_width + 1):
                line = rendering.Line((x * self.u_size, 0), (x * self.u_size, self.n_height * self.u_size))
                self.viewer.add_geom(line)
            for y in range(self.n_height + 1):
                line = rendering.Line((0, y * self.u_size), (self.n_width * self.u_size, y * self.u_size))
                self.viewer.add_geom(line)

            # 初始化智能体渲染对象
            self.agent_geoms = []
            for i, agent in enumerate(self.agents):
                agent_circle = rendering.make_circle(self.u_size / 4, 30, True)
                agent_color = i / self.n_agents
                agent_circle.set_color(agent_color, 0, 1 - agent_color)
                agent_trans = rendering.Transform()
                agent_circle.add_attr(agent_trans)
                self.viewer.add_geom(agent_circle)
                self.agent_geoms.append(agent_trans)

        # 动态更新格子颜色
        for i, (x, y) in enumerate([(x, y) for x in range(self.n_width) for y in range(self.n_height)]):
            rect = self.grid_geoms[i]
            grid = self.grids.get_grid(x, y)
            if grid.type == 0 and grid.value == 1:
                rect.set_color(0.5, 0.5, 0.5)  # 灰色
            elif grid.type == 0:
                rect.set_color(1, 1, 1)  # 白色
            else:
                rect.set_color(0, 0, 0)  # 黑色

        # 更新智能体位置
        for i, agent in enumerate(self.agents):
            x, y = self._state_to_xy(self.pos[agent])
            self.agent_geoms[i].set_translation((x + 0.5) * self.u_size, (y + 0.5) * self.u_size)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _state_to_xy(self, s):
        x = s % self.n_width
        y = int((s - x) / self.n_width)
        return x, y

    def _xy_to_state(self, x, y=None):
        if isinstance(x, int):
            assert (isinstance(y, int)), "incomplete Position info"
            return x + self.n_width * y
        elif isinstance(x, tuple):
            return x[0] + self.n_width * x[1]
        return -1  # 未知状态, unknow status

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def __del__(self):
        self.close()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Dict(
            {
                "tpm": Box(low=0, high=1, shape=(self.n_width * self.n_height,), dtype=np.float32),
                "state": Discrete(self.n_width * self.n_height),
                "_state": MultiDiscrete([self.n_width * self.n_height] * (self.n_agents - 1)),
                "direct": MultiDiscrete([3, 3], start=[-1, -1]),
                "_direct": MultiDiscrete([3, 3] * (self.n_agents - 1))
            }
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(3)

from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    env = MAGridWorldEnv(n_agents=3, n_width=10, n_height=10, u_size=60, default_reward=0.5, default_type=0)
    parallel_api_test(env, num_cycles=10000)