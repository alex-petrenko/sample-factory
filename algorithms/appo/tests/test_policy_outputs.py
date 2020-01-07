import copy
import pickle
import sys
from concurrent.futures.thread import ThreadPoolExecutor

import torch

from algorithms.appo.appo_utils import set_step_data, TaskType
from utils.timing import Timing
from utils.utils import log


def make_shared(data):
    if isinstance(data, dict):
        for k, v in data.items():
            make_shared(v)
    elif isinstance(data, (list, tuple)):
        for v in data:
            make_shared(v)
    elif isinstance(data, torch.Tensor):
        data.share_memory_()
    elif isinstance(data, (int, float, bool, TaskType)):
        pass
    else:
        raise RuntimeError(f'{type(data)}')


def _enq_task(actor_idx, split_idx, task, policy_outputs, output_tensors):
    for env_idx, agent_idx, output_idx in task:
        tensors_dict_key = actor_idx, split_idx, env_idx, agent_idx
        for key, value in policy_outputs.items():
            set_step_data(output_tensors[tensors_dict_key], key, value[output_idx])

    return 42


def _enqueue_policy_outputs(request_order, policy_outputs, output_tensors, executor, timing):
    output_idx = 0

    tasks_per_actor = dict()

    for actor_idx, split_idx, env_idx, agent_idx in request_order:
        key = (actor_idx, split_idx)
        if key in tasks_per_actor:
            tasks_per_actor[key].append((env_idx, agent_idx, output_idx))
        else:
            tasks_per_actor[key] = [(env_idx, agent_idx, output_idx)]

        output_idx += 1

    result = None
    for key, task in tasks_per_actor.items():
        actor_idx, split_idx = key
        _enq_task(actor_idx, split_idx, task, policy_outputs, output_tensors)

    # print(result.result())

    # for actor_idx, split_idx, env_idx, agent_idx in request_order:
    #     tensors_dict_key = actor_idx, split_idx, env_idx, agent_idx
    #
    #     for key, value in policy_outputs.items():
    #         set_step_data(output_tensors[tensors_dict_key], key, value[output_idx])
    #     set_step_data(output_tensors[tensors_dict_key], 'policy_version', 1)
    #
    #     output_idx += 1
    #
    #     outputs_ready.add((actor_idx, split_idx))

    # for actor_idx, split_idx in outputs_ready:
    #     advance_rollout_request = dict(split_idx=split_idx, policy_id=self.policy_id)
    #     self.actor_queues[actor_idx].put((TaskType.ROLLOUT_STEP, advance_rollout_request))


def main():
    with open('/tmp/policy_outputs3', 'rb') as fobj:
        data_item = pickle.load(fobj)

    timing = Timing()

    request_order, policy_outputs, output_tensors = data_item

    make_shared(output_tensors)

    executor = ThreadPoolExecutor(max_workers=10)

    for i in range(1304):
        log.debug('Progress: %d', i)

        outputs = copy.deepcopy(policy_outputs)

        with timing.add_time('cat_tens'):
            outputs_list = []
            sizes = []
            for key, value in outputs.items():
                value = value.float()
                if len(value.shape) == 1:
                    value = torch.unsqueeze(value, dim=1)
                outputs_list.append(value)
                sizes.append(value.shape[-1])

            outputs_cat = dict(all=torch.cat(outputs_list, dim=1))
            # outputs_split = torch.split(outputs_cat, split_size_or_sections=sizes, dim=1)

        with timing.add_time('enqueue'):
            _enqueue_policy_outputs(request_order, outputs_cat, output_tensors, executor, timing)

    executor.shutdown()

    log.info('Timing: %s', timing)


if __name__ == '__main__':
    sys.exit(main())
