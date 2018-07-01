from gym.envs.registration import register

register(
    id="fluids-v2",
    entry_point="gym_fluids.envs:FluidsEnv",
    timestep_limit=1000,
    nondeterministic=True,
    )
