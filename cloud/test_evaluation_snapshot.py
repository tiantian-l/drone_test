"""CPU-only lifecycle checks of the actual evaluation context implementation."""
import ast
import contextlib
from pathlib import Path
import threading
import types
import unittest


class Array:
    def __init__(self, value):
        self.value, self.deleted = value, False

    def copy(self):
        return Array(self.value)

    def delete(self):
        self.deleted = True


class EvaluationSnapshotTest(unittest.TestCase):
    def make_agent(self):
        path = Path(__file__).resolve().parents[1] / 'third_party/dreamerv3/embodied/jax/agent.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Agent')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluation')
        namespace = {
            'contextlib': contextlib,
            'internal': types.SimpleNamespace(move=lambda x, _: x),
            'jax': types.SimpleNamespace(tree=types.SimpleNamespace(
                map=lambda fn, xs: {k: fn(v) for k, v in xs.items()})),
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
        agent = types.SimpleNamespace(
            train_lock=threading.Lock(), policy_lock=threading.Lock(),
            params={'weight': Array(3)}, policy_keys=['weight'],
            policy_params={'weight': Array(1)}, pending_sync={'weight': Array(2)},
            policy_params_sharding=None, n_actions=99)
        agent.evaluation = types.MethodType(namespace['evaluation'], agent)
        return agent

    def test_snapshot_and_restore(self):
        agent = self.make_agent()
        original, pending = agent.policy_params, agent.pending_sync
        for _ in range(2):
            with agent.evaluation():
                snapshot = agent.policy_params['weight']
                self.assertEqual(snapshot.value, 3)
                self.assertIsNone(agent.pending_sync)
                self.assertEqual(agent._evaluation_counter, 0)
                agent._evaluation_counter = 42
                with self.assertRaises(RuntimeError):
                    with agent.evaluation():
                        pass
            self.assertTrue(snapshot.deleted)
            self.assertIs(agent.policy_params, original)
            self.assertIs(agent.pending_sync, pending)
            self.assertEqual(agent.n_actions, 99)
            self.assertFalse(agent.params['weight'].deleted)

    def test_exception_restores_training(self):
        agent = self.make_agent()
        original, pending = agent.policy_params, agent.pending_sync
        with self.assertRaises(ValueError):
            with agent.evaluation():
                raise ValueError('evaluation failed')
        self.assertIs(agent.policy_params, original)
        self.assertIs(agent.pending_sync, pending)
        self.assertIsNone(agent._evaluation_counter)


if __name__ == '__main__':
    unittest.main()
