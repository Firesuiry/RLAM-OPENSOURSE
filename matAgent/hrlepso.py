from pathlib import Path

from matAgent.hrlepso_base import *
from rl.DDPG.TF2_DDPG_Basic import DDPG
from env.HrelpsoBaseEnv import HrlepsoEnv


class HrlepsoSwarm(HrlepsoBaseSwarm):

    def __init__(self, n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic):
        super().__init__(n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic)
        model = config_dic.get('model',
                               r"D:\develop\swam\ieeeaccess - 副本 - 副本 - 副本 - 副本\rl\train0\ddpg_actor_episode100.h5")
        model_name = Path(model).name
        self.name = f'HRLEPSO-{model_name}'

        gym_env = HrlepsoEnv(show=False)
        ddpg = DDPG(gym_env, discrete=False, gamma=0, sigma=0.25, actor_units=(16,),
                    critic_units=(8, 16, 32))
        ddpg.load_actor(model)
        self.ddpg_actor = ddpg

    def run_once(self, action=np.zeros(10)):
        state = self.get_state()
        # print(f'state:{state}')
        action = self.ddpg_actor.policy(state)
        # print(f'action:{action}')
        if self.show:
            print(np.mean(np.abs(action)))
            print(action[:10])
        super(HrlepsoSwarm, self).run_once(action.numpy())


def fun2(x):
    x2 = np.power(x - 50, 2)
    fit = np.sum(x2, axis=-1)
    return fit


if __name__ == '__main__':
    model = r'''D:\develop\swam\ieeeaccess - 副本 - 副本 - 副本 - 副本\rl\train0\ddpg_actor_episode300.h5'''

    s = HrlepsoSwarm(1000, 40, True, fun2, 2, 100, -100, {'max_fes': 10000,
                                                          'model': model})
    s.run()
