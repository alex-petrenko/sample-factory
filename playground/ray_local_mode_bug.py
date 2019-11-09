import ray
ray.init(local_mode=True)

a = ray.put(42)

from copy import deepcopy
a_copy = deepcopy(a)
print(ray.get(a))
print(ray.get(a_copy))

#
# @ray.remote
# def remote_function(x):
#     obj = x['a']
#     return ray.get(obj)
#
#
# a = ray.put(42)
# d = {'a': a}
# result = remote_function.remote(d)
# print(ray.get(result))
