"""Resume entrypoint must be safe to re-import in spawned env workers."""
import importlib.abc
import multiprocessing
from pathlib import Path
import runpy
import sys
import unittest


SCRIPT = str(Path(__file__).with_name("resume_dynamic.py"))


class NoTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"elements", "ruamel", "dreamerv3"}:
            raise AssertionError(f"Worker import loaded training dependency: {fullname}")


def load_entrypoint(name):
    guard = NoTrainingImports()
    sys.meta_path.insert(0, guard)
    previous = sys.argv
    # A worker import must not parse the parent CLI or touch a log directory.
    sys.argv = [SCRIPT]
    try:
        namespace = runpy.run_path(SCRIPT, run_name=name)
        assert callable(namespace["main"])
    finally:
        sys.argv = previous
        sys.meta_path.remove(guard)


class ResumeImportTest(unittest.TestCase):
    def test_normal_import_has_no_training_side_effects(self):
        load_entrypoint("resume_dynamic")

    def test_spawn_reimport_has_no_training_side_effects(self):
        proc = multiprocessing.get_context("spawn").Process(
            target=load_entrypoint, args=("__mp_main__",))
        proc.start()
        try:
            proc.join(timeout=15)
            self.assertFalse(proc.is_alive(), "Worker hung during entrypoint import")
            self.assertEqual(proc.exitcode, 0)
        finally:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
            proc.close()


if __name__ == "__main__":
    unittest.main()
