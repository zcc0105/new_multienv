from gym.envs.registration import register

# Multiagent envs
# ----------------------------------------

register(
    id='BCIM_env',
    entry_point='multiagent.envs:simple_bidding',
    max_episode_steps=100,
)
