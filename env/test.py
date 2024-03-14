from gridworld import GridWorldEnv  # 从gridworld.py文件中导入GridWorldEnv类

def main():
    # 创建一个简单的格子世界环境实例
    env = GridWorldEnv(n_width=5,   # 环境宽度
                       n_height=5,  # 环境高度
                       u_size=40,   # 每个格子的像素大小
                       default_reward=-1,  # 默认奖励值
                       default_type=0)     # 默认类型(0表示可通过)

    # 设置起始位置和结束位置
    env.start = (0, 0)  # 起始位置
    env.ends = [(4, 4)]  # 结束位置，可以有多个，这里只设置了一个

    # 重置环境，准备开始新的一轮
    state = env.reset()

    # 模拟环境交互
    done = False
    total_reward = 0
    while not done:
        env.render()  # 渲染环境
        action = env.action_space.sample()  # 随机选择一个动作
        next_state, reward, done, _ = env.step(action)  # 执行动作，获取下一个状态、奖励和是否结束的标志

        print(f"Action: {action}, State: {next_state}, Reward: {reward}, Done: {done}")

        total_reward += reward

    print(f"Total reward: {total_reward}")

    env.render()  # 最后再渲染一次以显示最终状态
    env.close()  # 关闭环境

if __name__ == "__main__":
    main()
