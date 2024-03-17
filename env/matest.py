from gridworld import MAGridWorldEnv

def main():
    # 创建一个多智能体格子世界环境实例，假设有两个智能体
    env = MAGridWorldEnv(n_agents=2, n_width=10, n_height=10, u_size=60, default_reward=-1, default_type=0)

    # 重置环境，准备开始新的一轮
    states = env.reset()

    # 模拟环境交互
    done = [False] * env.n_agents
    total_rewards = [0] * env.n_agents
    step_count = 0

    while not all(done):
        env.render()  # 渲染环境
        actions = [env.agents[i].action_space.sample() for i in range(env.n_agents)]  # 每个智能体随机选择一个动作
        next_states, rewards, dones ,pos = env.step(actions)  # 执行动作，获取下一个状态、奖励和是否结束的标志

        for i in range(env.n_agents):
            print(f"Agent {i}, Action: {actions[i]}, State: {next_states[i]}, Reward: {rewards[i]}, Done: {dones[i]}, Pos: {pos[i]}")
            total_rewards[i] += rewards[i]
            done[i] = dones[i]

        step_count += 1
        if step_count >= 100:  # 限制最大步数，以防止无限循环
            break

    for i in range(env.n_agents):
        print(f"Agent {i}, Total reward: {total_rewards[i]}")

    env.render()  # 最后再渲染一次以显示最终状态
    env.close()  # 关闭环境

if __name__ == "__main__":
    main()
