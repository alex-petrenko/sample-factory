import time

from sample_factory.utils.dicts import list_of_dicts_to_dict_of_lists
from sample_factory.utils.network import is_udp_port_available
from sample_factory.utils.timing import Timing
from sample_factory.utils.utils import cores_for_worker_process, log


class TestUtils:
    def test_udp(self):
        is_udp_port_available(50301)

    def test_cpu_affinity(self):
        num_workers = 44
        cpu_count = 20

        for i in range(num_workers):
            cores = cores_for_worker_process(i, num_workers, cpu_count)
            if i < 40:
                assert cores == [i % cpu_count]
            elif i == 40:
                assert cores == [0, 1, 2, 3, 4]
            elif i == 41:
                assert cores == [5, 6, 7, 8, 9]
            elif i == 42:
                assert cores == [10, 11, 12, 13, 14]
            elif i == 43:
                assert cores == [15, 16, 17, 18, 19]

    def test_timing(self):
        t = Timing()

        with t.add_time("total"):
            with t.timeit("t1"):
                time.sleep(0.1)

            for i in range(3):
                with t.add_time("t2"):
                    time.sleep(0.05)
                    with t.add_time("t2.1"), t.add_time("t2.1.1"):
                        pass
                    with t.add_time("t2.2"):
                        pass
                    with t.add_time("t2.3"):
                        pass

            for i in range(4):
                with t.time_avg("t3"):
                    time.sleep(i % 2)
                    with t.time_avg("t3.1"):
                        pass

        log.debug(t.flat_str())
        log.debug(t)  # tree view

    def test_list_of_dicts_to_dict_of_lists(self):
        """Test list_of_dicts_to_dict_of_lists() with recursive dicts."""
        lt = [{"a": 1, "b": {"c": 2, "d": 3}}, {"a": 4, "b": {"c": 5, "d": 6}}]
        d = list_of_dicts_to_dict_of_lists(lt)
        assert d == {"a": [1, 4], "b": {"c": [2, 5], "d": [3, 6]}}
