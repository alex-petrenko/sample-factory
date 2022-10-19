from typing import Optional

try:
    import envpool
except ImportError as e:
    print(e)
    print("Trying to import envpool when it is not install. install with 'pip install envpool'")
import gym

from sf_examples.envpool_examples.envpool_wrappers import (
    BatchedRecordEpisodeStatistics,
    EnvPoolResetFixWrapper,
    EnvPoolTo5Tuple,
)

ATARI_W = ATARI_H = 84


class AtariSpec:
    def __init__(self, name, env_id, default_timeout=None):
        self.name = name
        self.env_id = env_id
        self.default_timeout = default_timeout
        self.has_timer = False


# Note NoFrameskip-v4 in gym[atari] is the same game configuration as -v5 in envpool
ATARI_ENVS = [
    AtariSpec("atari_alien", "Alien-v5"),
    AtariSpec("atari_amidar", "Amidar-v5"),
    AtariSpec("atari_assault", "Assault-v5"),
    AtariSpec("atari_asterix", "Asterix-v5"),
    AtariSpec("atari_asteroid", "Asteroids-v5"),
    AtariSpec("atari_atlantis", "Atlantis-v5"),
    AtariSpec("atari_bankheist", "BankHeist-v5"),
    AtariSpec("atari_battlezone", "BattleZone-v5"),
    AtariSpec("atari_beamrider", "BeamRider-v5"),
    AtariSpec("atari_berzerk", "Berzerk-v5"),
    AtariSpec("atari_bowling", "Bowling-v5"),
    AtariSpec("atari_boxing", "Boxing-v5"),
    AtariSpec("atari_breakout", "Breakout-v5"),
    AtariSpec("atari_centipede", "Centipede-v5"),
    AtariSpec("atari_choppercommand", "ChopperCommand-v5"),
    AtariSpec("atari_crazyclimber", "CrazyClimber-v5"),
    AtariSpec("atari_defender", "Defender-v5"),
    AtariSpec("atari_demonattack", "DemonAttack-v5"),
    AtariSpec("atari_doubledunk", "DoubleDunk-v5"),
    AtariSpec("atari_enduro", "Enduro-v5"),
    AtariSpec("atari_fishingderby", "FishingDerby-v5"),
    AtariSpec("atari_freeway", "Freeway-v5"),
    AtariSpec("atari_frostbite", "Frostbite-v5"),
    AtariSpec("atari_gopher", "Gopher-v5"),
    AtariSpec("atari_gravitar", "Gravitar-v5"),
    AtariSpec("atari_hero", "Hero-v5"),
    AtariSpec("atari_icehockey", "IceHockey-v5"),
    AtariSpec("atari_jamesbond", "Jamesbond-v5"),
    AtariSpec("atari_kangaroo", "Kangaroo-v5"),
    AtariSpec("atari_krull", "Krull-v5"),
    AtariSpec("atari_kongfumaster", "KungFuMaster-v5"),
    AtariSpec("atari_montezuma", "MontezumaRevenge-v5", default_timeout=18000),
    AtariSpec("atari_mspacman", "MsPacman-v5"),
    AtariSpec("atari_namethisgame", "NameThisGame-v5"),
    AtariSpec("atari_phoenix", "Phoenix-v5"),
    AtariSpec("atari_pitfall", "Pitfall-v5"),
    AtariSpec("atari_pong", "Pong-v5"),
    AtariSpec("atari_privateye", "PrivateEye-v5"),
    AtariSpec("atari_qbert", "Qbert-v5"),
    AtariSpec("atari_riverraid", "Riverraid-v5"),
    AtariSpec("atari_roadrunner", "RoadRunner-v5"),
    AtariSpec("atari_robotank", "Robotank-v5"),
    AtariSpec("atari_seaquest", "Seaquest-v5"),
    AtariSpec("atari_skiing", "Skiing-v5"),
    AtariSpec("atari_solaris", "Solaris-v5"),
    AtariSpec("atari_spaceinvaders", "SpaceInvaders-v5"),
    AtariSpec("atari_stargunner", "StarGunner-v5"),
    AtariSpec("atari_surround", "Surround-v5"),
    AtariSpec("atari_tennis", "Tennis-v5"),
    AtariSpec("atari_timepilot", "TimePilot-v5"),
    AtariSpec("atari_tutankham", "Tutankham-v5"),
    AtariSpec("atari_upndown", "UpNDown-v5"),
    AtariSpec("atari_venture", "Venture-v5"),
    AtariSpec("atari_videopinball", "VideoPinball-v5"),
    AtariSpec("atari_wizardofwor", "WizardOfWor-v5"),
    AtariSpec("atari_yarsrevenge", "YarsRevenge-v5"),
    AtariSpec("atari_zaxxon", "Zaxxon-v5"),
]


def atari_env_by_name(name):
    for cfg in ATARI_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception("Unknown Atari env")


def make_atari_env(env_name, cfg, env_config, render_mode: Optional[str] = None):
    assert cfg.num_envs_per_worker == 1, "when using envpool, set num_envs_per_worker=1 and use --env_agents="
    atari_spec = atari_env_by_name(env_name)

    env_kwargs = dict()

    # if hasattr(cfg, "render_mode"):
    #     env_kwargs["render_mode"] = cfg.render_mode

    if atari_spec.default_timeout is not None:
        # envpool max_episode_steps does not take into account frameskip. see https://github.com/sail-sg/envpool/issues/195
        env_kwargs["max_episode_steps"] = atari_spec.default_timeout // 4
    if env_config is not None:
        env_kwargs["seed"] = env_config.env_id

    env = envpool.make(
        atari_spec.env_id,
        env_type="gym",
        num_envs=cfg.env_agents,
        reward_clip=True,
        episodic_life=True,
        frame_skip=cfg.env_frameskip,
        **env_kwargs,
    )
    env = EnvPoolResetFixWrapper(env)
    env = BatchedRecordEpisodeStatistics(env, num_envs=cfg.env_agents)
    env = EnvPoolTo5Tuple(env)
    env.num_agents = cfg.env_agents
    return env
