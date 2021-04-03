from unittest import TestCase
from sample_factory.utils.utils import cores_for_worker_process
from sample_factory.utils.network import is_udp_port_available


class TestUtils(TestCase):
    def test_udp(self):
        is_udp_port_available(50301)

    def test_cpu_affinity(self):
        num_workers = 44
        cpu_count = 20

        for i in range(num_workers):
            cores = cores_for_worker_process(i, num_workers, cpu_count)
            if i < 40:
                self.assertEqual(cores, [i % cpu_count])
            elif i == 40:
                self.assertEqual(cores, [0, 1, 2, 3, 4])
            elif i == 41:
                self.assertEqual(cores, [5, 6, 7, 8, 9])
            elif i == 42:
                self.assertEqual(cores, [10, 11, 12, 13, 14])
            elif i == 43:
                self.assertEqual(cores, [15, 16, 17, 18, 19])


