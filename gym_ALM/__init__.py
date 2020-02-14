from gym.envs.registration import register

register(
    id = 'ALM-v0',
    entry_point = 'gym_ALM.envs:ALMEnv',
    kwargs = {'T': 80, 'rate': .06, 'hist_returns': True}
)
