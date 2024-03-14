import gridworld as gw
import gym
from gym import spaces
from gym.utils import seeding
from Agent import Agent
from gridworld import GridWorldEnv
import random
class MAGridWorldEnv(GridWorldEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, n_agents, **kwargs):
        super().__init__(**kwargs)  # 调用父类构造函数
        self.n_agents = n_agents  # 智能体数量
        self.agents = [Agent(self.start, agent_id) for agent_id in range(n_agents)]  # 创建多个智能体实例

        for agent in self.agents:
            agent.observation_space = spaces.Discrete(self.n_width * self.n_height)  # 设置观察空间
        self.state = []
        self.start = []
        self.action = []
        self.reward = []
        self.set_start()
        for i in range(n_agents):
            self.action.append(0)
            self.reward.append(0)
        self.reset()

    def set_start(self):
        for i, agent in enumerate(self.agents):
            self.start[i] = random.randint(0, self.n_width - 1), random.randint(0, self.n_height - 1)


    def reset(self):
        super().reset()
        for i, agent in enumerate(self.agents):
            agent.position = self.start[i]  # 重置所有智能体的位置
        # 可以添加更多的重置逻辑

    def step(self, actions):
        obs = []
        rewards = []
        dones = []
        for i, action in enumerate(actions):
            self.state[i] = self.agents[i].position  # 设置当前智能体的状态
            next_state, reward, done, info = super().step(action)
            n_x, n_y = self._state_to_xy(next_state)
            self.agents[i].position = self._state_to_xy(next_state)  # 更新智能体位置
            # 将智能体周围一圈格子全标识为已访问过
            for x in range(-1, 1):
                for y in range(-1, 1):
                    if():
                        self.grids.set_value(self.agents[i].position[0] + x, self.agents[i].position[1] + y, 1)

            obs[i] = self.agents[i].position
            rewards[i] = reward
            dones[i] = done

        # 根据你的需要添加更多逻辑，如处理智能体之间的交互
        return obs, rewards, dones

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.width, self.height)

            # 绘制格子
            for x in range(self.n_width):
                for y in range(self.n_height):
                    rect = rendering.FilledPolygon([
                        (x * self.u_size, y * self.u_size),
                        ((x + 1) * self.u_size, y * self.u_size),
                        ((x + 1) * self.u_size, (y + 1) * self.u_size),
                        (x * self.u_size, (y + 1) * self.u_size)
                    ])
                    rect.set_color(0.9, 0.9, 0.9)  # 灰色填充每个格子
                    self.viewer.add_geom(rect)

        # 为每个智能体绘制一个圆形表示其位置
        for i, agent in enumerate(self.agents):
            agent_circle = rendering.make_circle(self.u_size / 4, 30, True)
            agent_color = i / self.n_agents  # 根据智能体ID分配不同的颜色
            agent_circle.set_color(agent_color, 0, 1 - agent_color)  # 使用不同颜色以区分不同智能体
            self.viewer.add_geom(agent_circle)

            # 设置智能体位置的变换
            agent_trans = rendering.Transform()
            agent_circle.add_attr(agent_trans)

            # 根据智能体的位置更新圆形的位置
            agent_trans.set_translation((agent.position[0] + 0.5) * self.u_size,
                                        (agent.position[1] + 0.5) * self.u_size)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
