from gymnasium.envs.registration import register

register(
    id="Roller-v1",
    entry_point="src.env.roller:RollerEnv",
    max_episode_steps=1000,
)
