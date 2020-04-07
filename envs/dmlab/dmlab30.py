import collections


DMLAB_INSTRUCTIONS = 'INSTR'
DMLAB_VOCABULARY_SIZE = 1000
DMLAB_MAX_INSTRUCTION_LEN = 16


LEVEL_MAPPING = collections.OrderedDict([
    ('rooms_collect_good_objects_train', 'rooms_collect_good_objects_test'),
    ('rooms_exploit_deferred_effects_train', 'rooms_exploit_deferred_effects_test'),
    ('rooms_select_nonmatching_object', 'rooms_select_nonmatching_object'),
    ('rooms_watermaze', 'rooms_watermaze'),
    ('rooms_keys_doors_puzzle', 'rooms_keys_doors_puzzle'),
    ('language_select_described_object', 'language_select_described_object'),
    ('language_select_located_object', 'language_select_located_object'),
    ('language_execute_random_task', 'language_execute_random_task'),
    ('language_answer_quantitative_question', 'language_answer_quantitative_question'),
    ('lasertag_one_opponent_small', 'lasertag_one_opponent_small'),
    ('lasertag_three_opponents_small', 'lasertag_three_opponents_small'),
    ('lasertag_one_opponent_large', 'lasertag_one_opponent_large'),
    ('lasertag_three_opponents_large', 'lasertag_three_opponents_large'),
    ('natlab_fixed_large_map', 'natlab_fixed_large_map'),
    ('natlab_varying_map_regrowth', 'natlab_varying_map_regrowth'),
    ('natlab_varying_map_randomized', 'natlab_varying_map_randomized'),
    ('skymaze_irreversible_path_hard', 'skymaze_irreversible_path_hard'),
    ('skymaze_irreversible_path_varied', 'skymaze_irreversible_path_varied'),
    ('psychlab_arbitrary_visuomotor_mapping', 'psychlab_arbitrary_visuomotor_mapping'),
    ('psychlab_continuous_recognition', 'psychlab_continuous_recognition'),
    ('psychlab_sequential_comparison', 'psychlab_sequential_comparison'),
    ('psychlab_visual_search', 'psychlab_visual_search'),
    ('explore_object_locations_small', 'explore_object_locations_small'),
    ('explore_object_locations_large', 'explore_object_locations_large'),
    ('explore_obstructed_goals_small', 'explore_obstructed_goals_small'),
    ('explore_obstructed_goals_large', 'explore_obstructed_goals_large'),
    ('explore_goal_locations_small', 'explore_goal_locations_small'),
    ('explore_goal_locations_large', 'explore_goal_locations_large'),
    ('explore_object_rewards_few', 'explore_object_rewards_few'),
    ('explore_object_rewards_many', 'explore_object_rewards_many'),
])

DMLAB30_LEVELS = tuple(LEVEL_MAPPING.keys())


HUMAN_SCORES = {
    'rooms_collect_good_objects_test': 10,
    'rooms_exploit_deferred_effects_test': 85.65,
    'rooms_select_nonmatching_object': 65.9,
    'rooms_watermaze': 54,
    'rooms_keys_doors_puzzle': 53.8,
    'language_select_described_object': 389.5,
    'language_select_located_object': 280.7,
    'language_execute_random_task': 254.05,
    'language_answer_quantitative_question': 184.5,
    'lasertag_one_opponent_small': 12.65,
    'lasertag_three_opponents_small': 18.55,
    'lasertag_one_opponent_large': 18.6,
    'lasertag_three_opponents_large': 31.5,
    'natlab_fixed_large_map': 36.9,
    'natlab_varying_map_regrowth': 24.45,
    'natlab_varying_map_randomized': 42.35,
    'skymaze_irreversible_path_hard': 100,
    'skymaze_irreversible_path_varied': 100,
    'psychlab_arbitrary_visuomotor_mapping': 58.75,
    'psychlab_continuous_recognition': 58.3,
    'psychlab_sequential_comparison': 39.5,
    'psychlab_visual_search': 78.5,
    'explore_object_locations_small': 74.45,
    'explore_object_locations_large': 65.65,
    'explore_obstructed_goals_small': 206,
    'explore_obstructed_goals_large': 119.5,
    'explore_goal_locations_small': 267.5,
    'explore_goal_locations_large': 194.5,
    'explore_object_rewards_few': 77.7,
    'explore_object_rewards_many': 106.7,
}

RANDOM_SCORES = {
    'rooms_collect_good_objects_test': 0.073,
    'rooms_exploit_deferred_effects_test': 8.501,
    'rooms_select_nonmatching_object': 0.312,
    'rooms_watermaze': 4.065,
    'rooms_keys_doors_puzzle': 4.135,
    'language_select_described_object': -0.07,
    'language_select_located_object': 1.929,
    'language_execute_random_task': -5.913,
    'language_answer_quantitative_question': -0.33,
    'lasertag_one_opponent_small': -0.224,
    'lasertag_three_opponents_small': -0.214,
    'lasertag_one_opponent_large': -0.083,
    'lasertag_three_opponents_large': -0.102,
    'natlab_fixed_large_map': 2.173,
    'natlab_varying_map_regrowth': 2.989,
    'natlab_varying_map_randomized': 7.346,
    'skymaze_irreversible_path_hard': 0.1,
    'skymaze_irreversible_path_varied': 14.4,
    'psychlab_arbitrary_visuomotor_mapping': 0.163,
    'psychlab_continuous_recognition': 0.224,
    'psychlab_sequential_comparison': 0.129,
    'psychlab_visual_search': 0.085,
    'explore_object_locations_small': 3.575,
    'explore_object_locations_large': 4.673,
    'explore_obstructed_goals_small': 6.76,
    'explore_obstructed_goals_large': 2.61,
    'explore_goal_locations_small': 7.66,
    'explore_goal_locations_large': 3.14,
    'explore_object_rewards_few': 2.073,
    'explore_object_rewards_many': 2.438,
}

RANDOM_POLICY_EPISODE_LEN = {
    'rooms_collect_good_objects_train': 3600,
    'rooms_exploit_deferred_effects_train': 3600,
    'rooms_select_nonmatching_object': 720,
    'rooms_watermaze': 7200,
    'rooms_keys_doors_puzzle': 3468,
    'language_select_described_object': 3600,
    'language_select_located_object': 7200,
    'language_execute_random_task': 7200,
    'language_answer_quantitative_question': 3600,
    'lasertag_one_opponent_small': 14400,
    'lasertag_three_opponents_small': 14400,
    'lasertag_one_opponent_large': 14400,
    'lasertag_three_opponents_large': 14400,
    'natlab_fixed_large_map': 7200,
    'natlab_varying_map_regrowth': 7200,
    'natlab_varying_map_randomized': 7200,
    'skymaze_irreversible_path_hard': 3600,
    'skymaze_irreversible_path_varied': 3372,
    'psychlab_arbitrary_visuomotor_mapping': 18000,
    'psychlab_continuous_recognition': 18000,
    'psychlab_sequential_comparison': 18000,
    'psychlab_visual_search': 9000,
    'explore_object_locations_small': 5400,
    'explore_object_locations_large': 7200,
    'explore_obstructed_goals_small': 5400,
    'explore_obstructed_goals_large': 7200,
    'explore_goal_locations_small': 5400,
    'explore_goal_locations_large': 7200,
    'explore_object_rewards_few': 5400,
    'explore_object_rewards_many': 7200,
}


# this is how many episodes are required for one billion frames of training on DMLab-30
# Used for level cache generation. Only levels that require level cache generation are listed.
# here 1B = 250M samples * frameskip, frameskip=4

# the actual value will of course be different since episode lengths change as policy improves
# this is also under assumption that the agent is trained for the same number of envs on every level
DMLAB30_APPROX_NUM_EPISODES_PER_BILLION_FRAMES = {
    'rooms_keys_doors_puzzle': 11200,
    'lasertag_one_opponent_small': 2400,
    'lasertag_three_opponents_small': 2400,
    'lasertag_one_opponent_large': 2400,
    'lasertag_three_opponents_large': 2400,
    'skymaze_irreversible_path_hard': 11200,
    'skymaze_irreversible_path_varied': 13500,
    'explore_object_locations_small': 6200,
    'explore_object_locations_large': 4700,
    'explore_obstructed_goals_small': 6200,
    'explore_obstructed_goals_large': 4700,
    'explore_goal_locations_small': 6200,
    'explore_goal_locations_large': 4700,
    'explore_object_rewards_few': 6200,
    'explore_object_rewards_many': 4700
}

DMLAB30_LEVELS_THAT_USE_LEVEL_CACHE = tuple(DMLAB30_APPROX_NUM_EPISODES_PER_BILLION_FRAMES.keys())


def dmlab30_num_envs():
    num_envs = len(tuple(LEVEL_MAPPING.keys()))
    return num_envs


def dmlab30_level_name_to_level(level_name):
    return f'contributed/dmlab30/{level_name}'
