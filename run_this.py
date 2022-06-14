from field_env import Flow_Field
from field_env import point
from RL_brain import DeepQNetwork


MAX_EPISODE=300
def start_swim():
    step = 0
    for episode in range(MAX_EPISODE):
        # initial observation
        observation = env.reset()

        while True:
            
            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # 训练结束
    print('train finished!')
    env.destroy()


if __name__ == "__main__":
    # 强化学习推进策略
    env = Flow_Field()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    env.after(100, start_swim)
    env.mainloop()
