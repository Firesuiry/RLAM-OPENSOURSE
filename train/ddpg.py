from rl.DDPG.TF2_DDPG_Basic import DDPG
import numpy as np


# print('?')


def get_ddpg_object(
        env,
        discrete=False,
        use_priority=False,
        lr_actor=1e-5,
        lr_critic=1e-3,
        actor_units=(24, 16),
        critic_units=(24, 16),
        noise='norm',
        sigma=0.15,
        tau=0.125,
        gamma=0.85,
        batch_size=64,
        memory_cap=100000):
    return DDPG(env, discrete=discrete, memory_cap=memory_cap, actor_units=(16, 32, 32, 32, 64, 64),
                critic_units=(8, 16, 32, 32, 16, 8), use_priority=True, lr_critic=lr_critic, lr_actor=lr_actor)


def train():
    from env.TestpsoEnv import TestpsoEnv
    print('train')
    gym_env = TestpsoEnv(show=False)
    try:
        # Ensure action bound is symmetric
        assert (gym_env.action_space.high == -gym_env.action_space.low)
        is_discrete = False
        print('Continuous Action Space')
    except AttributeError:
        is_discrete = True
        print('Discrete Action Space')
    # ddpg = DDPG(gym_env, discrete=is_discrete, memory_cap=10000000, actor_units=(16, 32, 64),
    #             critic_units=(8, 16, 32), use_priority=True, lr_critic=1e-7, lr_actor=1e-9)
    ddpg = DDPG(gym_env, discrete=is_discrete, memory_cap=10000000, actor_units=(16,),
                critic_units=(8, 16, 32), use_priority=True, lr_critic=1e-7, lr_actor=1e-9)
    ddpg.train(max_episodes=10000, max_steps=1000)


def test():
    from env.PsoEnv import PsoEnv

    gym_env = PsoEnv(show=False)
    try:
        # Ensure action bound is symmetric
        assert (gym_env.action_space.high == -gym_env.action_space.low)
        is_discrete = False
        print('Continuous Action Space')
    except AttributeError:
        is_discrete = True
        print('Discrete Action Space')

    ddpg = DDPG(gym_env, discrete=is_discrete)
    # ddpg.load_critic("ddpg_critic_episode124.h5")
    # ddpg.load_actor("actor01.h5")

    step_nums = []
    for i in range(10):
        reward, step_num = ddpg.test()
        step_nums.append(step_num)
        print('step:{}'.format(step_num))
    print('trained mean:{}'.format(np.mean(step_nums)))

    step_nums = []
    for i in range(10):
        step_num = gym_env.test()
        step_nums.append(step_num)
        print('step:{}'.format(step_num))
    print('origin mean:{}'.format(np.mean(step_nums)))

    # ddpg.train(max_episodes=1000)


if __name__ == "__main__":
    print('start')
    train()
