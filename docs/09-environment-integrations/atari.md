# Atari
<video width="800" controls autoplay><source src="https://huggingface.co/datasets/edbeeching/sample_factory_videos/resolve/main/atari_grid_57_60s.mp4" type="video/mp4"></video>

### Installation

Install Sample Factory with Atari dependencies with PyPI:

```
pip install sample-factory[atari]
```

### Running Experiments

Run Atari experiments with the scripts in `sf_examples.atari`.

The default parameters have been chosen to match CleanRL's configuration (see reports below) and are not tuned for throughput.
(TODO: also provide parameters that result in the fastest training).
 

To train a model in the `BreakoutNoFrameskip-v4` environment:

```
python -m sf_examples.atari.train_atari --algo=APPO --env=atari_breakout --experiment="Experiment Name"
```

To visualize the training results, use the `enjoy_atari` script:

```
python -m sf_examples.atari.enjoy_atari --algo=APPO --env=atari_breakout --experiment="Experiment Name"
```

Multiple experiments can be run in parallel with the launcher module. `atari_envs` is an example launcher script that runs atari envs with 4 seeds. 

```
python -m sample_factory.launcher.run --run=sf_examples.atari.experiments.atari_envs --backend=processes --max_parallel=8  --pause_between=1 --experiments_per_gpu=10000 --num_gpus=1
```

### List of Supported Environments

Specify the environment to run with the `--env` command line parameter. The following Atari v4 environments are supported out of the box.
Various APPO models trained on Atari environments are uploaded to the HuggingFace Hub. The models have all been trained for 2 billion steps with 3 seeds per experiment. Videos of the agents after training can be found on the HuggingFace Hub.

| Atari Command Line Parameter | Atari Environment name         | Model Checkpooints                                                                                 |
| ---------------------------- | ------------------------------ | -------------------------------------------------------------------------------------------------- |
| atari_alien                  | AlienNoFrameskip-v4            | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_alien_1111)          |
| atari_amidar                 | AmidarNoFrameskip-v4           | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_amidar_1111)         |
| atari_assault                | AssaultNoFrameskip-v4          | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_assault_1111)        |
| atari_asterix                | AsterixNoFrameskip-v4          | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_asterix_1111)        |
| atari_asteroid               | AsteroidsNoFrameskip-v4        | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_asteroid_1111)       |
| atari_atlantis               | AtlantisNoFrameskip-v4         | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_atlantis_1111)       |
| atari_bankheist              | BankHeistNoFrameskip-v4        | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_bankheist_1111)      |
| atari_battlezone             | BattleZoneNoFrameskip-v4       | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_battlezone_1111)     |
| atari_beamrider              | BeamRiderNoFrameskip-v4        | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_beamrider_1111)      |
| atari_berzerk                | BerzerkNoFrameskip-v4          | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_berzerk_1111)        |
| atari_bowling                | BowlingNoFrameskip-v4          | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_bowling_1111)        |
| atari_boxing                 | BoxingNoFrameskip-v4           | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_boxing_1111)         |
| atari_breakout               | BreakoutNoFrameskip-v4         | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_breakout_1111)       |
| atari_centipede              | CentipedeNoFrameskip-v4        | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_centipede_1111)      |
| atari_choppercommand         | ChopperCommandNoFrameskip-v4   | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_choppercommand_1111) |
| atari_crazyclimber           | CrazyClimberNoFrameskip-v4     | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_crazyclimber_1111)   |
| atari_defender               | DefenderNoFrameskip-v4         | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_defender_1111)       |
| atari_demonattack            | DemonAttackNoFrameskip-v4      | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_demonattack_1111)    |
| atari_doubledunk             | DoubleDunkNoFrameskip-v4       | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_doubledunk_1111)     |
| atari_enduro                 | EnduroNoFrameskip-v4           | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_enduro_1111)         |
| atari_fishingderby           | FishingDerbyNoFrameskip-v4     | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_fishingderby_1111)   |
| atari_freeway                | FreewayNoFrameskip-v4          | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_freeway_1111)        |
| atari_frostbite              | FrostbiteNoFrameskip-v4        | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_frostbite_1111)      |
| atari_gopher                 | GopherNoFrameskip-v4           | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_gopher_1111)         |
| atari_gravitar               | GravitarNoFrameskip-v4         | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_gravitar_1111)       |
| atari_hero                   | HeroNoFrameskip-v4             | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_hero_1111)           |
| atari_icehockey              | IceHockeyNoFrameskip-v4        | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_icehockey_1111)      |
| atari_jamesbond              | JamesbondNoFrameskip-v4        | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_jamesbond_1111)      |
| atari_kangaroo               | KangarooNoFrameskip-v4         | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_kangaroo_1111)       |
| atari_krull                  | KrullNoFrameskip-v4            | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_krull_1111)          |
| atari_kongfumaster           | KungFuMasterNoFrameskip-v4     | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_kongfumaster_1111)   |
| atari_montezuma              | MontezumaRevengeNoFrameskip-v4 | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_montezuma_1111)      |
| atari_mspacman               | MsPacmanNoFrameskip-v4         | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_mspacman_1111)       |
| atari_namethisgame           | NameThisGameNoFrameskip-v4     | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_namethisgame_1111)   |
| atari_phoenix                | PhoenixNoFrameskip-v4          | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_phoenix_1111)        |
| atari_pitfall                | PitfallNoFrameskip-v4          | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_pitfall_1111)        |
| atari_pong                   | PongNoFrameskip-v4             | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_pong_1111)           |
| atari_privateye              | PrivateEyeNoFrameskip-v4       | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_privateye_1111)      |
| atari_qbert                  | QbertNoFrameskip-v4            | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_qbert_1111)          |
| atari_riverraid              | RiverraidNoFrameskip-v4        | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_riverraid_1111)      |
| atari_roadrunner             | RoadRunnerNoFrameskip-v4       | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_roadrunner_1111)     |
| atari_robotank               | RobotankNoFrameskip-v4         | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_robotank_1111)       |
| atari_seaquest               | SeaquestNoFrameskip-v4         | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_seaquest_1111)       |
| atari_skiing                 | SkiingNoFrameskip-v4           | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_skiing_1111)         |
| atari_solaris                | SolarisNoFrameskip-v4          | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_solaris_1111)        |
| atari_spaceinvaders          | SpaceInvadersNoFrameskip-v4    | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_spaceinvaders_1111)  |
| atari_stargunner             | StarGunnerNoFrameskip-v4       | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_stargunner_1111)     |
| atari_surround               | SurroundNoFrameskip-v4         | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_surround_1111)       |
| atari_tennis                 | TennisNoFrameskip-v4           | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_tennis_1111)         |
| atari_timepilot              | TimePilotNoFrameskip-v4        | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_timepilot_1111)      |
| atari_tutankham              | TutankhamNoFrameskip-v4        | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_tutankham_1111)      |
| atari_upndown                | UpNDownNoFrameskip-v4          | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_upndown_1111)        |
| atari_venture                | VentureNoFrameskip-v4          | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_venture_1111)        |
| atari_videopinball           | VideoPinballNoFrameskip-v4     | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_videopinball_1111)   |
| atari_wizardofwor            | WizardOfWorNoFrameskip-v4      | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_wizardofwor_1111)    |
| atari_yarsrevenge            | YarsRevengeNoFrameskip-v4      | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_yarsrevenge_1111)    |
| atari_zaxxon                 | ZaxxonNoFrameskip-v4           | [🤗 Hub Atari-2B checkpoints](https://huggingface.co/edbeeching/atari_2B_atari_zaxxon_1111)         |


### Reports

- Sample Factory was benchmarked on Atari against CleanRL and Baselines. Sample Factory was able to achieve similar sample efficiency as CleanRL and Baselines using the same parameters.
    - https://wandb.ai/wmfrank/atari-benchmark/reports/Atari-Sample-Factory2-Baselines-CleanRL--VmlldzoyMzEyNjIw
