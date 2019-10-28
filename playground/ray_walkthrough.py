import time

import ray


if not ray.is_initialized():
    ray.init(local_mode=True)


# A regular Python function.
def regular_function():
    return 1

# A Ray remote function.
@ray.remote
def remote_function():
    print('Remote!')
    return 1


object_id = remote_function.remote()
result = ray.get(object_id)
print(object_id, result)


def test(value):
    print(value)


@ray.remote(num_return_vals=2)
def remote_chain_function(value):
    print('Value:', value)
    return value + 1, str(value)


y1_id = remote_function.remote()
# assert ray.get(y1_id) == 1

test(y1_id)
# chained_id, str_id = remote_chain_function.remote(y1_id)
# print(chained_id, str_id)
# res1, res2 = ray.get(chained_id), ray.get(str_id)
# print(res1, res2)


y = 11
object_id = ray.put(y)

print(ray.get(object_id))


print('Test...')


@ray.remote
class TestActor:
    def __init__(self):
        self.a = []
        self.ready = False

    def sum(self, x):
        self.a.append(x)
        while len(self.a) < 10:
            time.sleep(0.01)
        print('Collected 10 items', self.a)


@ray.remote
class Worker:
    def __init__(self, worker_id):
        self.id = worker_id

    def task(self, test_actor_):
        print('Calling sum for', self.id)
        task_id = test_actor_.sum.remote(self.id)
        print('Waiting for', task_id)
        ray.get(task_id)
        print('Task finished for', self.id)


workers = [Worker.remote(i) for i in range(3)]
test_actor = TestActor.remote()


for w in workers:
    w.task.remote(test_actor)


ray.shutdown()
