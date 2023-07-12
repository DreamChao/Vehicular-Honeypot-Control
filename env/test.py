from env.VehicularHoneypotEnv import VehicularHoneypotEnv
import numpy as np
import time

if __name__ == '__main__':
    env = VehicularHoneypotEnv()
    obs = env.reset()
    # for i in range(5):
    #     print("\n-----------------")
    #     print(i)
    #     score = 0
    #     action = np.random.randint(4)  # 用随机数产生策略
    #     observation = env.reset()
    #     for t in range(200):
    #         observation, reward, done, info = env.step(action)
    #         score += reward  # 记录奖励
    #         time.sleep(0.01)
    #     print('Policy Score', score)
    # env.close()


    step = 0
    score = 0
    for t in range(10):
        step += 1
        print("\n")
        print("-----------step:", step)
        print("------")

        action = np.random.randint(4)
        print("action:", action)
        obs, reward, done, _ = env.step(action)
        score += reward
        if done:
            break
        print(f"state : {obs}, reward : {reward}")
        print("是否结束:", done)
    print(score)