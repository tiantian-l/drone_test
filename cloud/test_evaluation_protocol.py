"""Dependency-free checks of the shared evaluation protocol."""
import importlib.util
from pathlib import Path
import unittest
import contextlib
import sys
import types
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / 'third_party/dreamerv3/embodied/run/evaluation.py'
spec = importlib.util.spec_from_file_location('evaluation_protocol', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ProtocolTest(unittest.TestCase):
    def test_fresh_driver_each_cycle_and_cleanup(self):
        instances = []
        class Scalar(float):
            ndim = 0
        class Driver:
            def __init__(self, factories, parallel):
                self.callbacks, self.closed = [], False
                instances.append(self)
            def on_step(self, callback):
                self.callbacks.append(callback)
            def reset(self, init):
                init(1)
            def __call__(self, policy, episodes_per_env):
                for slot in range(episodes_per_env):
                    policy(None, None)
                    tran = {'is_first': True, 'is_last': True}
                    for key, value in dict(eval_map_slot=slot, eval_map_seed=190000+slot,
                                           success=1, collision=0, crash=0, timeout=0).items():
                        tran['log/last/' + key] = Scalar(value)
                    for callback in self.callbacks:
                        callback(tran, 0)
            def close(self):
                self.closed = True
        agent = types.SimpleNamespace(evaluation=contextlib.nullcontext,
            init_policy=lambda n: None, policy=lambda *a, **kw: None)
        with patch.dict(sys.modules, {'embodied': types.SimpleNamespace(Driver=Driver)}):
            first, _ = module.evaluate(agent, [lambda: None], 8)
            second, _ = module.evaluate(agent, [lambda: None], 8)
            self.assertEqual(first, second)
            def fail(*args):
                raise RuntimeError('callback failed')
            with self.assertRaises(RuntimeError):
                module.evaluate(agent, [lambda: None], 8, callbacks=(fail,))
        self.assertEqual(len(instances), 3)
        self.assertTrue(all(x.closed for x in instances))

    def records(self):
        return [dict(worker=w, eval_map_slot=s, eval_map_seed=190000+s*16+w,
                     success=int(s < 4), collision=int(s >= 4), crash=0, timeout=0)
                for s in range(8) for w in range(16)]

    def test_128_maps_and_groups(self):
        result = module.summarize(self.records())
        self.assertEqual(result['episodes'], 128)
        self.assertEqual(result['unique_maps'], 128)
        self.assertEqual(result['success'], .5)
        self.assertEqual(result['original_success'], 1)
        self.assertEqual(result['added_success'], 0)

    def test_duplicate_rejected(self):
        rows = self.records()
        rows[-1]['eval_map_seed'] = rows[0]['eval_map_seed']
        with self.assertRaises(ValueError):
            module.summarize(rows)

    def test_rotated_order_rejected(self):
        rows = self.records()
        rows[0], rows[16] = rows[16], rows[0]
        with self.assertRaises(ValueError):
            module.summarize(rows)


if __name__ == '__main__':
    unittest.main()
