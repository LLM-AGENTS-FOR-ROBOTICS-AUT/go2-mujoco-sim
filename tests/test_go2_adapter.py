"""Go2Adapter against a fake unitree_sdk2py: command mapping, state conversion, watchdog, LiDAR parsing."""
import math
import struct
import sys
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeSportClient:
    instances = []

    def __init__(self, *args):
        self.calls = []
        FakeSportClient.instances.append(self)

    def SetTimeout(self, t):
        pass

    def Init(self):
        pass

    def __getattr__(self, name):
        if name[0].isupper():
            return lambda *args: self.calls.append((name, args)) or 0
        raise AttributeError(name)


class FakeVideo:
    def SetTimeout(self, t):
        pass

    def Init(self):
        pass


class FakeSubscriber:
    handlers = {}

    def __init__(self, topic, kind):
        self.topic = topic

    def Init(self, handler, depth):
        FakeSubscriber.handlers[self.topic] = handler

    def Close(self):
        pass


def install_fake_sdk():
    root = types.ModuleType("unitree_sdk2py")
    modules = {
        "unitree_sdk2py": root,
        "unitree_sdk2py.core": types.ModuleType("core"),
        "unitree_sdk2py.core.channel": SimpleNamespace(ChannelFactoryInitialize=lambda domain, iface: None,
                                                       ChannelSubscriber=FakeSubscriber),
        "unitree_sdk2py.go2": types.ModuleType("go2"),
        "unitree_sdk2py.go2.sport": types.ModuleType("sport"),
        "unitree_sdk2py.go2.sport.sport_client": SimpleNamespace(SportClient=FakeSportClient),
        "unitree_sdk2py.go2.video": types.ModuleType("video"),
        "unitree_sdk2py.go2.video.video_client": SimpleNamespace(VideoClient=FakeVideo),
        "unitree_sdk2py.idl": types.ModuleType("idl"),
        "unitree_sdk2py.idl.sensor_msgs": types.ModuleType("sm"),
        "unitree_sdk2py.idl.sensor_msgs.msg": types.ModuleType("smm"),
        "unitree_sdk2py.idl.sensor_msgs.msg.dds_": SimpleNamespace(PointCloud2_=object),
        "unitree_sdk2py.idl.unitree_go": types.ModuleType("ug"),
        "unitree_sdk2py.idl.unitree_go.msg": types.ModuleType("ugm"),
        "unitree_sdk2py.idl.unitree_go.msg.dds_": SimpleNamespace(SportModeState_=object, LowState_=object),
    }
    sys.modules.update(modules)


def sport_state(mode=1, x=1., y=2., yaw=.5, vx_world=0., vy_world=0., yaw_speed=0., roll=0., pitch=0.):
    return SimpleNamespace(imu_state=SimpleNamespace(rpy=[roll, pitch, yaw]), velocity=[vx_world, vy_world, 0.],
                           yaw_speed=yaw_speed, position=[x, y, .3], body_height=.31, mode=mode, error_code=0,
                           gait_type=1)


def cloud(points, frame="odom"):
    data = b"".join(struct.pack("<fff", *p) for p in points)
    fields = [SimpleNamespace(name=n, offset=4 * i, datatype=7, count=1) for i, n in enumerate("xyz")]
    return SimpleNamespace(fields=fields, data=list(data), point_step=12, is_bigendian=False,
                           header=SimpleNamespace(frame_id=frame))


class Go2AdapterTests(unittest.TestCase):
    def setUp(self):
        install_fake_sdk()
        from go2.adapters.go2_sdk import Go2Adapter
        FakeSportClient.instances.clear()
        self.adapter = Go2Adapter("en7")
        FakeSubscriber.handlers.clear()
        import threading
        def feed():
            while not FakeSubscriber.handlers.get("rt/sportmodestate"):
                time.sleep(.01)
            FakeSubscriber.handlers["rt/sportmodestate"](sport_state())
        threading.Thread(target=feed, daemon=True).start()
        self.adapter.connect()
        self.client = FakeSportClient.instances[-1]

    def tearDown(self):
        self.adapter.close()

    def push(self, **kwargs):
        FakeSubscriber.handlers["rt/sportmodestate"](sport_state(**kwargs))

    def test_requires_arming_and_reports_connection(self):
        self.assertTrue(self.adapter.requires_arming)
        self.assertTrue(self.adapter.connected)

    def test_state_converts_world_velocity_to_body_frame(self):
        self.push(mode=3, yaw=math.pi / 2, vx_world=0., vy_world=.5)
        s = self.adapter.state()
        self.assertAlmostEqual(s.vx, .5, places=6)
        self.assertAlmostEqual(s.vy, 0., places=6)
        self.assertEqual(s.posture, "walking")
        self.push(mode=3)
        self.assertEqual(self.adapter.state().posture, "standing", "locomotion mode at rest is standing")
        self.push(mode=5)
        self.assertEqual(self.adapter.state().posture, "lying")
        self.push(mode=1, roll=1.4)
        self.assertEqual(self.adapter.state().posture, "fallen")

    def test_command_mapping(self):
        cases = {"stand_up": "StandUp", "lie_down": "StandDown", "sit": "Sit", "recovery_stand": "RecoveryStand",
                 "damp": "Damp", "hello": "Hello", "stretch": "Stretch", "content": "Content", "heart": "Heart",
                 "scrape": "Scrape", "dance1": "Dance1", "dance2": "Dance2", "front_flip": "FrontFlip",
                 "back_flip": "BackFlip", "left_flip": "LeftFlip", "front_jump": "FrontJump", "front_pounce": "FrontPounce"}
        for name, method in cases.items():
            self.client.calls.clear()
            self.adapter.sport(name)
            self.assertEqual(self.client.calls[0][0], method, name)
        self.client.calls.clear()
        self.adapter.sport("euler", roll=.1, pitch=-.2, yaw=.3)
        self.assertEqual(self.client.calls, [("Euler", (.1, -.2, .3))])
        self.client.calls.clear()
        self.adapter.sport("speed_level", level=-1)
        self.assertEqual(self.client.calls, [("SpeedLevel", (-1,))])

    def test_special_modes_end_on_gait_change(self):
        self.adapter.sport("handstand")
        self.assertEqual(self.client.calls[-1], ("HandStand", (True,)))
        self.client.calls.clear()
        self.adapter.sport("gait", gait="classic")
        self.assertEqual(self.client.calls, [("HandStand", (False,)), ("ClassicWalk", (True,))])

    def test_stand_from_sitting_uses_rise_sit(self):
        self.push(mode=10)
        self.adapter.sport("stand")
        self.assertEqual(self.client.calls[-1][0], "RiseSit")

    def test_move_watchdog_stops_when_refreshes_stop(self):
        self.adapter.move(.3, 0, 0)
        self.assertEqual(self.client.calls[-1], ("Move", (.3, 0., 0.)))
        time.sleep(.7)
        self.assertEqual(self.client.calls[-1], ("StopMove", ()))

    def test_lidar_odom_cloud_becomes_base_frame_without_own_body(self):
        self.push(x=1., y=2., yaw=math.pi / 2)
        FakeSubscriber.handlers["rt/utlidar/cloud_deskewed"](cloud([(1., 4., .3), (1.05, 2.05, .2)]))
        points, state = self.adapter.lidar()
        self.assertEqual(len(points), 1, "points inside the body box are dropped")
        self.assertAlmostEqual(points[0][0], 2., places=4)   # 2 m straight ahead
        self.assertAlmostEqual(points[0][1], 0., places=4)


if __name__ == "__main__":
    unittest.main()
