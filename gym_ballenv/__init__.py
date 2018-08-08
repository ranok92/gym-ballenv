from gym.envs.registration import register


register(
		id = 'gymball-v0',
		entry_point = 'gym_ballenv.envs:BallEnv',
		timestep_limit = 1000,
		reward_threshold = 100.0,
		nondeterministic = True,

		)

