"""
General GridWorld Environment

Author: Qiang Ye
Date: July 22, 2017


License: MIT
"""

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random
from Agent import Agent
import time
import pyglet


class Grid(object):  # 代表单个格子
    def __init__(self, x: int = None,
                 y: int = None,
                 type: int = 0,
                 reward: int = 0.0,
                 value: float = 0.0):  # value, for possible future usage
        self.x = x  # coordinate x
        self.y = y
        self.type = type  # Type (0:empty；1:obstacle or boundary)
        self.reward = reward  # instant reward for an agent entering this grid cell
        self.value = value  # the value of this grid cell, 标识是否被探索过
        self.name = None  # name of this grid.
        self._update_name()

    def _update_name(self):
        self.name = "X{0}-Y{1}".format(self.x, self.y)

    def __str__(self):
        return "name:{4}, x:{0}, y:{1}, type:{2}, value{3}".format(self.x,
                                                                   self.y,
                                                                   self.type,
                                                                   self.reward,
                                                                   self.value,
                                                                   self.name
                                                                   )


class GridMatrix(object):
    '''格子矩阵，通过不同的设置，模拟不同的格子世界环境
    '''

    def __init__(self, n_width: int,  # defines the number of cells horizontally
                 n_height: int,  # vertically
                 default_type: int = 0,  # default cell type
                 default_reward: float = 0.0,  # default instant reward
                 default_value: float = 0.0  # default value
                 ):
        self.grids = None
        self.n_height = n_height
        self.n_width = n_width
        self.len = n_width * n_height
        self.default_reward = default_reward
        self.default_value = default_value
        self.default_type = default_type
        self.reset()

    def reset(self):
        self.grids = []
        for x in range(self.n_height):
            for y in range(self.n_width):
                self.grids.append(Grid(x,
                                       y,
                                       self.default_type,
                                       self.default_reward,
                                       self.default_value))

    def get_grid(self, x, y=None):
        '''get a grid information
        args: represented by x,y or just a tuple type of x
        return: grid object
        '''
        xx, yy = None, None
        if isinstance(x, int):
            xx, yy = x, y
        elif isinstance(x, tuple):
            xx, yy = x[0], x[1]

        if not (xx >= 0 and yy >= 0 and xx < self.n_width and yy < self.n_height):
            print("xx:{0}, yy:{1}".format(xx, yy) + "异常!!!!!!!!!!!!!!!!!!!!!")
        assert (xx >= 0 and yy >= 0 and xx < self.n_width and yy < self.n_height), \
            "coordinates should be in reasonable range"
        index = yy * self.n_width + xx
        return self.grids[index]

    def set_reward(self, x, y, reward):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.reward = reward
        else:
            raise ("grid doesn't exist")

    def set_value(self, x, y, value):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.value = value
            # 打印格子的坐标和修改成的值
            # print("设置x:" + str(x) + ", y:" + str(y) + "格子为" + str(value))
        else:
            raise ("grid doesn't exist")

    def set_type(self, x, y, type):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.type = type
        else:
            raise ("grid doesn't exist")

    def get_reward(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.reward

    def get_value(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.value

    def get_type(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.type

    def is_existed(self, x, y):
        # print("查询x:" + str(x) + " ,y:" + str(y) + "是否存在")
        if 0 <= x < self.n_width and 0 <= y < self.n_height:
            # print("x:"+str(x)+" y:"+str(y)+"存在")
            return True
        else:
            return False


class GridWorldEnv(gym.Env):
    '''格子世界环境，可以模拟各种不同的格子世界
    '''
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, n_width: int = 10,
                 n_height: int = 7,
                 u_size=40,
                 default_reward: float = 0,
                 default_type=0,
                 windy=False):
        self.u_size = u_size  # size for each cell (pixels)
        self.n_width = n_width  # width of the env calculated by number of cells.
        self.n_height = n_height  # height...
        self.width = u_size * n_width  # scenario width (pixels)
        self.height = u_size * n_height  # height
        self.default_reward = default_reward
        self.default_type = default_type
        self._adjust_size()

        self.grids = GridMatrix(n_width=self.n_width,
                                n_height=self.n_height,
                                default_reward=self.default_reward,
                                default_type=self.default_type,
                                default_value=0.0)
        self.reward = 0  # for rendering
        self.action = None  # for rendering
        self.windy = windy  # whether this is a windy environment

        # 0,1,2,3,4 represent left, right, up, down, -, five moves.
        self.action_space = spaces.Discrete(4)
        # 观察空间由low和high决定
        self.observation_space = spaces.Discrete(self.n_height * self.n_width)
        # 坐标原点为左下角，这个pyglet是一致的, left-bottom corner is the position of (0,0)
        # 通过设置起始点、终止点以及特殊奖励和类型的格子可以构建各种不同类型的格子世界环境
        # 比如：随机行走、汽车租赁、悬崖行走等David Silver公开课中的示例
        self.ends = [(7, 3)]  # 终止格子坐标，可以有多个, goal cells position list
        self.start = (0, 3)  # 起始格子坐标，只有一个, start cell position, only one start position
        self.types = []  # 特殊种类的格子在此设置。[(3,2,1)]表示(3,2)处值为1.
        # special type of cells, (x,y,z) represents in position(x,y) the cell type is z
        self.rewards = []  # 特殊奖励的格子在此设置，终止格子奖励0, special reward for a cell
        self.refresh_setting()
        self.viewer = None  # 图形接口对象
        self._seed()  # 产生一个随机子
        # self.reset()

    def _adjust_size(self):
        '''调整场景尺寸适合最大宽度、高度不超过800
        '''
        pass

    def _seed(self, seed=None):
        # 产生一个随机化时需要的种子，同时返回一个np_random对象，支持后续的随机化生成操作
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # time.sleep(5)
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))

        self.action = action  # action for rendering
        old_x, old_y = self._state_to_xy(self.state)
        new_x, new_y = old_x, old_y

        # wind effect:
        # 有风效果，其数字表示个体离开(而不是进入)该格子时朝向别的方向会被吹偏离的格子数
        # this effect is just used for the windy env in David Silver's youtube video.
        if self.windy:
            if new_x in [3, 4, 5, 8]:
                new_y += 1
            elif new_x in [6, 7]:
                new_y += 2

        if action == 0:
            new_x -= 1  # left
        elif action == 1:
            new_x += 1  # right
        elif action == 2:
            new_y += 1  # up
        elif action == 3:
            new_y -= 1  # down

        elif action == 4:
            new_x, new_y = new_x - 1, new_y - 1
        elif action == 5:
            new_x, new_y = new_x + 1, new_y - 1
        elif action == 6:
            new_x, new_y = new_x + 1, new_y - 1
        elif action == 7:
            new_x, new_y = new_x + 1, new_y + 1
        # boundary effect
        if new_x < 0: new_x = 0
        if new_x >= self.n_width: new_x = self.n_width - 1
        if new_y < 0: new_y = 0
        if new_y >= self.n_height: new_y = self.n_height - 1

        # wall effect, obstacles or boundary.
        # 类型为1的格子为障碍格子，不可进入
        if self.grids.get_type(new_x, new_y) == 1:
            # new_x, new_y = old_x, old_y
            done = 1
        self.reward = self.grids.get_reward(new_x, new_y)

        done = self._is_end_state(new_x, new_y)
        self.state = self._xy_to_state(new_x, new_y)
        # 提供格子世界所有的信息在info内
        info = {"x": new_x, "y": new_y, "grids": self.grids}
        return self.state, self.reward, done, info

    # 将状态变为横纵坐标, set status into an one-axis coordinate value
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

    def refresh_setting(self):
        '''用户在使用该类创建格子世界后可能会修改格子世界某些格子类型或奖励值
        的设置，修改设置后通过调用该方法使得设置生效。
        '''
        for x, y, r in self.rewards:
            self.grids.set_reward(x, y, r)
        for x, y, t in self.types:
            self.grids.set_type(x, y, t)

    def reset(self):
        self.state = self._xy_to_state(self.start)
        return self.state

        # 判断是否是终止状态

    def _is_end_state(self, x, y=None):
        if y is not None:
            xx, yy = x, y
        elif isinstance(x, int):
            xx, yy = self._state_to_xy(x)
        else:
            assert (isinstance(x, tuple)), "incomplete coordinate values"
            xx, yy = x[0], x[1]
        for end in self.ends:
            if xx == end[0] and yy == end[1]:
                return True
        return False

    # 图形化界面, Graphic UI
    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        zero = (0, 0)
        u_size = self.u_size
        m = 2  # gaps between two cells

        # 如果还没有设定屏幕对象，则初始化整个屏幕具备的元素。
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.width, self.height)

            # 在Viewer里绘制一个几何图像的步骤如下：
            # the following steps just tells how to render an shape in the environment.
            # 1. 建立该对象需要的数据本身
            # 2. 使用rendering提供的方法返回一个geom对象
            # 3. 对geom对象进行一些对象颜色、线宽、线型、变换属性的设置（有些对象提供一些个
            #    性化的方法来设置属性，具体请参考继承自这些Geom的对象），这其中有一个重要的
            #    属性就是变换属性，
            #    该属性负责对对象在屏幕中的位置、渲染、缩放进行渲染。如果某对象
            #    在呈现时可能发生上述变化，则应建立关于该对象的变换属性。该属性是一个
            #    Transform对象，而一个Transform对象，包括translate、rotate和scale
            #    三个属性，每个属性都由以np.array对象描述的矩阵决定。
            # 4. 将新建立的geom对象添加至viewer的绘制对象列表里，如果在屏幕上只出现一次，
            #    将其加入到add_onegeom(）列表中，如果需要多次渲染，则将其加入add_geom()
            # 5. 在渲染整个viewer之前，对有需要的geom的参数进行修改，修改主要基于该对象
            #    的Transform对象
            # 6. 调用Viewer的render()方法进行绘制
            ''' 绘制水平竖直格子线，由于设置了格子之间的间隙，可不用此段代码
            for i in range(self.n_width+1):
                line = rendering.Line(start = (i*u_size, 0), 
                                      end =(i*u_size, u_size*self.n_height))
                line.set_color(0.5,0,0)
                self.viewer.add_geom(line)
            for i in range(self.n_height):
                line = rendering.Line(start = (0, i*u_size),
                                      end = (u_size*self.n_width, i*u_size))
                line.set_color(0,0,1)
                self.viewer.add_geom(line)
            '''

            # 绘制格子, draw cells
            for x in range(self.n_width):
                for y in range(self.n_height):
                    v = [(x * u_size + m, y * u_size + m),
                         ((x + 1) * u_size - m, y * u_size + m),
                         ((x + 1) * u_size - m, (y + 1) * u_size - m),
                         (x * u_size + m, (y + 1) * u_size - m)]

                    rect = rendering.FilledPolygon(v)
                    r = self.grids.get_reward(x, y) / 10
                    if r < 0:
                        rect.set_color(0.9 - r, 0.9 + r, 0.9 + r)
                    elif r > 0:
                        rect.set_color(0.3, 0.5 + r, 0.3)
                    else:
                        rect.set_color(0.9, 0.9, 0.9)
                    self.viewer.add_geom(rect)
                    # 绘制边框, draw frameworks
                    v_outline = [(x * u_size + m, y * u_size + m),
                                 ((x + 1) * u_size - m, y * u_size + m),
                                 ((x + 1) * u_size - m, (y + 1) * u_size - m),
                                 (x * u_size + m, (y + 1) * u_size - m)]
                    outline = rendering.make_polygon(v_outline, False)
                    outline.set_linewidth(3)

                    if self._is_end_state(x, y):
                        # 给终点方格添加金黄色边框, give end state cell a yellow outline.
                        outline.set_color(0.9, 0.9, 0)
                        self.viewer.add_geom(outline)
                    if self.start[0] == x and self.start[1] == y:
                        outline.set_color(0.5, 0.5, 0.8)
                        self.viewer.add_geom(outline)
                    if self.grids.get_type(x, y) == 1:  # 障碍格子用深灰色表示, obstacle cells are with gray color
                        rect.set_color(0.3, 0.3, 0.3)
                    else:
                        pass
            # 绘制个体, draw agent
            self.agent = rendering.make_circle(u_size / 4, 30, True)
            self.agent.set_color(1.0, 1.0, 0.0)
            self.viewer.add_geom(self.agent)
            self.agent_trans = rendering.Transform()
            self.agent.add_attr(self.agent_trans)

        # 更新个体位置 update position of an agent
        x, y = self._state_to_xy(self.state)
        self.agent_trans.set_translation((x + 0.5) * u_size, (y + 0.5) * u_size)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


def LargeGridWorld():
    '''10*10的一个格子世界环境，设置参照：
    http://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_td.html
    '''
    env = GridWorldEnv(n_width=10,
                       n_height=10,
                       u_size=40,
                       default_reward=0,
                       default_type=0,
                       windy=False)
    env.start = (0, 9)
    env.ends = [(5, 4)]
    env.types = [(4, 2, 1), (4, 3, 1), (4, 4, 1), (4, 5, 1), (4, 6, 1), (4, 7, 1),
                 (1, 7, 1), (2, 7, 1), (3, 7, 1), (4, 7, 1), (6, 7, 1), (7, 7, 1),
                 (8, 7, 1)]
    env.rewards = [(3, 2, -1), (3, 6, -1), (5, 2, -1), (6, 2, -1), (8, 3, -1),
                   (8, 4, -1), (5, 4, 1), (6, 4, -1), (5, 5, -1), (6, 5, -1)]
    env.refresh_setting()
    return env


def SimpleGridWorld():
    '''无风10*7的格子，设置参照： David Silver强化学习公开课视频 第3讲
    '''
    env = GridWorldEnv(n_width=10,
                       n_height=7,
                       u_size=60,
                       default_reward=-1,
                       default_type=0,
                       windy=False)
    env.start = (0, 3)
    env.ends = [(7, 3)]
    env.rewards = [(7, 3, 1)]
    env.refresh_setting()
    return env


def WindyGridWorld():
    '''有风10*7的格子，设置参照： David Silver强化学习公开课视频 第5讲
    '''
    env = GridWorldEnv(n_width=10,
                       n_height=7,
                       u_size=60,
                       default_reward=-1,
                       default_type=0,
                       windy=True)
    env.start = (0, 3)
    env.ends = [(7, 3)]
    env.rewards = [(7, 3, 1)]

    env.refresh_setting()
    return env


def RandomWalk():
    '''随机行走示例环境
    '''
    env = GridWorldEnv(n_width=7,
                       n_height=1,
                       u_size=80,
                       default_reward=0,
                       default_type=0,
                       windy=False)
    env.action_space = spaces.Discrete(2)  # left or right
    env.start = (3, 0)
    env.ends = [(6, 0), (0, 0)]
    env.rewards = [(6, 0, 1)]
    env.refresh_setting()
    return env


def CliffWalk():
    '''悬崖行走格子世界环境
    '''
    env = GridWorldEnv(n_width=12,
                       n_height=4,
                       u_size=60,
                       default_reward=-1,
                       default_type=0,
                       windy=False)
    env.action_space = spaces.Discrete(4)  # left or right
    env.start = (0, 0)
    env.ends = [(11, 0)]
    # env.rewards=[]
    # env.types = [(5,1,1),(5,2,1)]
    for i in range(10):
        env.rewards.append((i + 1, 0, -100))
        env.ends.append((i + 1, 0))
    env.refresh_setting()
    return env


def SkullAndTreasure():
    '''骷髅与钱币示例，解释随机策略的有效性 David Silver 强化学习公开课第六讲 策略梯度
    Examples of Skull and Money explained the necessity and effectiveness
    '''
    env = GridWorldEnv(n_width=5,
                       n_height=2,
                       u_size=60,
                       default_reward=-1,
                       default_type=0,
                       windy=False)
    env.action_space = spaces.Discrete(4)  # left or right
    env.start = (0, 1)
    env.ends = [(2, 0)]
    env.rewards = [(0, 0, -100), (2, 0, 100), (4, 0, -100)]
    env.types = [(1, 0, 1), (3, 0, 1)]
    env.refresh_setting()
    return env


class MAGridWorldEnv():
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
        self.start = [[0, 0] for i in range(n_agents)]
        self.targets = [[3, 2] , [5, 1],[7,4]]  # 目标位置
        self.agents = [Agent(self.start, agent_id) for agent_id in range(n_agents)]  # 创建多个智能体实例
        for agent in self.agents:
            agent.observation_space = spaces.Discrete(self.n_width * self.n_height)  # 设置观察空间
        self.state = [0] * n_agents  # 用于存储每个智能体的状态

        self.obs = [[0] * (self.n_width * self.n_height) for i in range(n_agents)] * n_agents  # 用于存储每个智能体的观察
        # self.start = [0, 0]*n_agents
        # print(self.state)
        self.action = []
        self.reward = []

        # self.directions = [[0, 1]]*n_agents
        # 方向向量，用于计算智能体的移动。默认朝上
        self.set_start()
        # self.set_
        self.viewer = None  # 图形接口对象
        self.agent_geoms = []  # Initialize the list for agent rendering objects
        self._seed()  # 产生一个随机子
        for i in range(n_agents):
            self.action.append(0)
            self.reward.append(0)
        self.reset()

    def _seed(self, seed=None):
        # 产生一个随机化时需要的种子，同时返回一个np_random对象，支持后续的随机化生成操作
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_start(self):
        # 清空start列表
        for i, agent in enumerate(self.agents):
            # 随机设置每个智能体的起始位置
            # self.start.append((random.randint(0, self.n_width - 1), random.randint(0, self.n_height - 1)))
            self.start[i] = (random.randint(0, self.n_width - 1), random.randint(0, self.n_height - 1))
    def set_target(self,target_list):
        for i in range(self.n_targets):
            self.grids.set_type(target_list[i][0], target_list[i][1], 2)
            self.grids.set_reward(target_list[i][0], target_list[i][1], 10)



    def reset(self):
        next_state = []
        for i, agent in enumerate(self.agents):
            agent.position = self.start[i]  # 重置所有智能体的位置
            self.state[i] = self._xy_to_state(
                self.start[i][0],
                self.start[i][1])  # 设置当前智能体的状态
            # agent.direct = [0, 1]  # 设置朝向
            # next_state[i].append(self.obs[i])
            # next_state[i].append(self.state[i])
            # next_state[i].append(agent.direct)
            # _state=[]
            # for j , agent in enumerate(self.agents):
            #     if(i!=j)_state.append(next_state[i])
        return self.obs
        # 可以添加更多的重置逻辑

    def step(self, actions):
        rewards = [0] * self.n_agents
        dones = [0] * self.n_agents
        pos = [0, 0] * self.n_agents

        # 定义动作
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

        for i, action in enumerate(actions):
            # time.sleep(0.5)
            done = 0
            reward = 0
            self.action[i] = actions[i]  # action for rendering
            old_x, old_y = self._state_to_xy(self.state[i])
            new_x, new_y = old_x, old_y

            # if action == FORWARD:
            #     # 沿当前朝向前进一格
            #     new_x, new_y = new_x + self.agents[i].direct[0], new_y + self.agents[i].direct[1]
            #
            # elif action in direction_changes:
            #     # 计算新位置
            #     trans = self.agents[i].direct
            #     if action == LEFT_FORWARD:
            #         print("Agent " + str(i) + " is turning left")
            #         trans = go_left(trans)
            #     elif action == RIGHT_FORWARD:
            #         print("Agent " + str(i) + " is turning right")
            #         trans = go_right(trans)
            #     new_x, new_y = new_x + trans[0], new_y + trans[1]
            #     # 计算新朝向
            #     change = direction_changes[action]
            #     self.agents[i].direct = [self.agents[i].direct[1] * change[0], self.agents[i].direct[0] * change[1]]

            if action in direction_changes:
                if action == LEFT_FORWARD:print("Agent " + str(i) + " is turning left")
                elif action == RIGHT_FORWARD:print("Agent " + str(i) + " is turning right")
                # 计算新朝向
                change = direction_changes[action]
                self.agents[i].direct = [self.agents[i].direct[1] * change[0], self.agents[i].direct[0] * change[1]]

            # 沿当前朝向前进一格
            new_x, new_y = new_x + self.agents[i].direct[0], new_y + self.agents[i].direct[1]

            # boundary effect
            if new_x < 0: new_x = 0
            if new_x >= self.n_width: new_x = self.n_width - 1
            if new_y < 0: new_y = 0
            if new_y >= self.n_height: new_y = self.n_height - 1
            # if new_x < 0: done = 1
            # if new_x >= self.n_width: done = 1
            # if new_y < 0: done = 1
            # if new_y >= self.n_height: done = 1

            pos[i] = (new_x, new_y)
            # wall effect, obstacles or boundary.
            # 类型为1的格子为障碍格子，不可进入
            if self.grids.is_existed(new_x,new_y) or self.grids.get_type(new_x, new_y) == 1:
                # new_x, new_y = old_x, old_y
                done = 1

            obs = []
            # 处理智能体周围一圈格子
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if self.grids.is_existed(new_x + x, new_y + y):
                        # and self.grids.get_type(new_x + x, new_y + y) == 0):
                        self.grids.set_value(new_x + x, new_y + y,
                                             1)  # 设置为已访问
                        self.reward[i] += self.grids.get_reward(new_x + x, new_y + y)  # 获取奖励
                        self.grids.set_reward(new_x + x, new_y + y, 0)  # 获取奖励后将奖励置0
            for x in range(self.n_width):
                for y in range(self.n_height):
                    obs.append(self.grids.get_value(x, y))  # 获取观察

            self.state[i] = self._xy_to_state(new_x, new_y)
            self.agents[i].position = (new_x, new_y)
            self.obs[i] = obs
            dones[i] = done

        print("all stepped!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # 根据你的需要添加更多逻辑，如处理智能体之间的交互
        return self.obs, self.reward, dones, pos

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
            for i ,agent in range(self.n_agents):
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
            x, y = agent.position
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


if __name__ == "__main__":
    env = GridWorldEnv()
    print("hello")
    env.reset()
    nfs = env.observation_space
    nfa = env.action_space
    print("nfs:%s; nfa:%s" % (nfs, nfa))
    print(env.observation_space)
    print(env.action_space)
    print(env.state)
    env.render()
    # x = input("press any key to exit")
    for _ in range(20000):
        env.render()
        a = env.action_space.sample()
        state, reward, isdone, info = env.step(a)
        print("{0}, {1}, {2}, {3}".format(a, reward, isdone, info))

    print("env closed")
