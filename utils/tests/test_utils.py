from unittest import TestCase

from utils.network import is_udp_port_available


class TestNetwork(TestCase):
    def test_udp(self):
        is_udp_port_available(50301)


