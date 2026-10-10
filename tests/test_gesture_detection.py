import sys
import types
import unittest
from pathlib import Path

# Stub the TensorFlow modules that are imported by the ASL recognizer so the app can
# be imported for gesture-level tests even without an ML environment installed.
tensorflow = types.ModuleType('tensorflow')
keras = types.ModuleType('tensorflow.keras')
layers = types.ModuleType('tensorflow.keras.layers')

class DummyBatchNormalization:
    @classmethod
    def from_config(cls, config):
        return cls()

class DummyDense:
    @classmethod
    def from_config(cls, config):
        return cls()

layers.BatchNormalization = DummyBatchNormalization
layers.Dense = DummyDense
keras.layers = layers
tensorflow.keras = keras
sys.modules['tensorflow'] = tensorflow
sys.modules['tensorflow.keras'] = keras
sys.modules['tensorflow.keras.layers'] = layers

ROOT = Path(__file__).resolve().parents[1]
MUSIC_PLAYER_DIR = ROOT / 'Music Player'
if str(MUSIC_PLAYER_DIR) not in sys.path:
    sys.path.insert(0, str(MUSIC_PLAYER_DIR))

import music_player


class FakeLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class FakeHand:
    def __init__(self, coords):
        self.landmark = [FakeLandmark(*point) for point in coords]


def build_hand(thumb_tip, thumb_ip, finger_tips, finger_pips, wrist=(0.5, 0.9, 0.0)):
    coords = [FakeLandmark(*wrist)]
    coords.extend([FakeLandmark(0.0, 0.0, 0.0) for _ in range(1, 21)])

    coord_map = {
        0: wrist,
        3: thumb_ip,
        4: thumb_tip,
        5: finger_pips['index_mcp'],
        6: finger_pips['index_pip'],
        7: finger_pips['index_dip'],
        8: finger_tips['index_tip'],
        9: finger_pips['middle_mcp'],
        10: finger_pips['middle_pip'],
        11: finger_pips['middle_dip'],
        12: finger_tips['middle_tip'],
        13: finger_pips['ring_mcp'],
        14: finger_pips['ring_pip'],
        15: finger_pips['ring_dip'],
        16: finger_tips['ring_tip'],
        17: finger_pips['pinky_mcp'],
        18: finger_pips['pinky_pip'],
        19: finger_pips['pinky_dip'],
        20: finger_tips['pinky_tip'],
    }

    for index, point in coord_map.items():
        coords[index] = FakeLandmark(*point)
    return FakeHand([(lm.x, lm.y, lm.z) for lm in coords])


class GestureDetectionTests(unittest.TestCase):
    def test_pause_is_not_read_as_search_toggle(self):
        hand = build_hand(
            thumb_tip=(0.42, 0.60, 0.0),
            thumb_ip=(0.46, 0.68, 0.0),
            finger_tips={
                'index_tip': (0.52, 0.83, 0.0),
                'middle_tip': (0.60, 0.82, 0.0),
                'ring_tip': (0.68, 0.85, 0.0),
                'pinky_tip': (0.74, 0.86, 0.0),
            },
            finger_pips={
                'index_mcp': (0.52, 0.70, 0.0),
                'index_pip': (0.52, 0.63, 0.0),
                'index_dip': (0.52, 0.72, 0.0),
                'middle_mcp': (0.60, 0.71, 0.0),
                'middle_pip': (0.60, 0.62, 0.0),
                'middle_dip': (0.60, 0.73, 0.0),
                'ring_mcp': (0.68, 0.72, 0.0),
                'ring_pip': (0.68, 0.61, 0.0),
                'ring_dip': (0.68, 0.74, 0.0),
                'pinky_mcp': (0.74, 0.73, 0.0),
                'pinky_pip': (0.74, 0.64, 0.0),
                'pinky_dip': (0.74, 0.75, 0.0),
            },
        )

        self.assertIn(music_player.recognize_gesture(hand), (None, 'pause'))

    def test_clear_pause_fist_is_detected_as_pause(self):
        hand = build_hand(
            thumb_tip=(0.40, 0.72, 0.0),
            thumb_ip=(0.44, 0.70, 0.0),
            finger_tips={
                'index_tip': (0.58, 0.82, 0.0),
                'middle_tip': (0.62, 0.83, 0.0),
                'ring_tip': (0.66, 0.84, 0.0),
                'pinky_tip': (0.70, 0.85, 0.0),
            },
            finger_pips={
                'index_mcp': (0.58, 0.72, 0.0),
                'index_pip': (0.58, 0.62, 0.0),
                'index_dip': (0.58, 0.74, 0.0),
                'middle_mcp': (0.62, 0.73, 0.0),
                'middle_pip': (0.62, 0.63, 0.0),
                'middle_dip': (0.62, 0.75, 0.0),
                'ring_mcp': (0.66, 0.74, 0.0),
                'ring_pip': (0.66, 0.64, 0.0),
                'ring_dip': (0.66, 0.76, 0.0),
                'pinky_mcp': (0.70, 0.75, 0.0),
                'pinky_pip': (0.70, 0.65, 0.0),
                'pinky_dip': (0.70, 0.77, 0.0),
            },
        )

        self.assertEqual(music_player.recognize_gesture(hand), 'pause')


if __name__ == '__main__':
    unittest.main()
