from MAGridWorldEnv import MAGridWorldEnv

# def main():
#     # 创建一个多智能体格子世界环境实例，假设有两个智能体
#     env = MAGridWorldEnv(n_agents=3, n_width=10, n_height=10, u_size=60, default_reward=0.5, default_type=0)
#     env.grids.set_type(3, 4, 1)  # 设置(3, 4)位置为障碍物
#     # 重置环境，准备开始新的一轮
#     states = env.reset()
#
#     # 模拟环境交互
#     done = [False] * env.n_agents
#     total_rewards = [0] * env.n_agents
#     step_count = 0
#
#     while 1:
#         env.render()  # 渲染环境
#         actions = [env.agents[i].action_space.sample() for i in range(env.n_agents)]  # 每个智能体随机选择一个动作
#         next_states, rewards, dones ,pos = env.step(actions)  # 执行动作，获取下一个状态、奖励和是否结束的标志
#
#         for i in range(env.n_agents):
#             # print(f"Agent {i}, Action: {actions[i]}, State: {next_states[i]}, Reward: {rewards[i]}, Done: {dones[i]}, Pos: {pos[i]}")
#
#             total_rewards[i] += rewards[i]
#             done[i] = dones[i]
#
#         step_count += 1
#         if step_count >= 100:
#             break
#
#     for i in range(env.n_agents):
#         print(f"Agent {i}, Total reward: {total_rewards[i]}")
#
#     env.render()  # 最后再渲染一次以显示最终状态
#     env.close()  # 关闭环境
#
# if __name__ == "__main__":
#     main()
# Import the environment class
from MAGridWorldEnv import MAGridWorldEnv
from pettingzoo.test import parallel_api_test

# Create an instance of the environment
env = MAGridWorldEnv(n_agents=3, n_width=10, n_height=10, u_size=60, default_reward=0.5, default_type=0)

# Run the parallel API test
parallel_api_test(env, num_cycles=10000)

# Example of manual stepping and rendering the environment
env.reset()
for agent in env.agents:
    action = env.action_space(agent).sample()  # Randomly sample an action
    observation, reward, termination, truncation, info = env.step({agent: action})
    # env.render()
    if termination or truncation:
        break

# Close the environment
env.close()

