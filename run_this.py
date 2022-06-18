from field_env import Flow_Field
from RL_brain import DeepQNetwork
import numpy as np

MAX_EPISODE = 2
def start_swim():
    action_space = ['plus5', 'plus10', 'minus5', 'minus10', 'zero']
    pointer = open('total_reward.txt','w+')
    for episode in range(MAX_EPISODE):
        # 初始物体状态
        vertex = open('swimmer.vertex', 'r')
        n = int(vertex.readline())
        Lag = []
        for i in range(n):
            tmp = vertex.readline().split()
            Lag.append([float(tmp[0]), float(tmp[1])])
        x = 0.0
        y = 0.0
        n = len(Lag)
        for i in range(n):
            x += Lag[i][0]
            y += Lag[i][1]
        x = x/float(n)
        y = y/float(n)
        observation = [x, y, 0, 0, 0]
        observation = np.array(observation)
        # 初始环境
        env = Flow_Field()
        env.episode = episode
        done = False
        step = 0 #记录步数d
        while not done:
            # RL choose action based on observation
            action = RL.choose_action(observation)
            print('========action = ', action_space[action], '==========\n')
            #action = random.choice(env.action_space)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action_space[action])

            env.total_reward += reward

            RL.store_transition(observation, action, reward, observation_)

            if (step > 5) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = np.array(observation_)

            step += 1
        print('=========================Episode ', episode, ' Total Reward = ', env.total_reward, '================================\n')
        pointer.write('Episode ' + str(episode) + ' Total Reward = ' + str(env.total_reward) + '\n')
    pointer.close()
    # 训练结束
    print('train finished!')
    # env.destroy()


if __name__ == "__main__":
    # 强化学习推进策略
    n_actions = 5
    n_features = 5
    RL = DeepQNetwork(n_actions, n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    start_swim()
    RL.plot_cost()
"""

"""