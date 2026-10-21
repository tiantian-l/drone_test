import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from render_failed_episodes import run_batch


class BatchTest(unittest.TestCase):
    def test_failure_selection_continue_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'diagnostics';source.mkdir()
            (source/'run.json').write_text(json.dumps(dict(status='complete',mode='full',episodes=3)))
            for seed,success in [(10,1),(12,0),(11,0)]:
                (source/f'{seed}.episode.json').write_text(json.dumps(dict(seed=seed,outcome=dict(success=success))))
            output=root/'media';calls=[]
            def execute(command, **kwargs):
                seed=int(command[command.index('--seed')+1]); path=Path(command[command.index('--output')+1]);calls.append((seed,path.suffix))
                if seed==11 and path.suffix=='.png': return SimpleNamespace(returncode=1)
                path.write_bytes(b'fixture')
                if path.suffix=='.mp4': path.with_suffix('.camera.json').write_text('{"frames":2}')
                return SimpleNamespace(returncode=0)
            with patch('render_failed_episodes.subprocess.run',side_effect=execute):
                self.assertEqual(run_batch(source,output),1)
                self.assertEqual(calls,[(11,'.png'),(11,'.mp4'),(12,'.png'),(12,'.mp4')])
                calls.clear()
                self.assertEqual(run_batch(source,output),1)
                self.assertEqual(calls,[(11,'.png')])
            report=[json.loads(x) for x in (output/'batch.jsonl').read_text().splitlines()]
            self.assertEqual(sum(r['status']=='skipped' for r in report),3)

if __name__=='__main__':unittest.main()
