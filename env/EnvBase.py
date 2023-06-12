class Env:
    def __init__(self, obs_shape, action_shape, action_low, action_high):
        self.observation_space = ObsShape(obs_shape)
        self.action_space = ActionSpace(action_shape, action_low, action_high)

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self):
        pass


class ObsShape:
    def __init__(self, shape):
        self.shape = shape


class ActionSpace:
    def __init__(self, shape, low, high):
        self.shape, self.low, self.high = shape, low, high
