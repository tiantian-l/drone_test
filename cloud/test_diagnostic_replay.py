import tempfile
from pathlib import Path
import unittest
import numpy as np
from replay_diagnostic_episode import camera, frame_indices, cylinder_faces, render
from test_diagnostics_analysis import AnalysisTest


class ReplayTests(unittest.TestCase):
    def test_camera_encloses_bounds_and_obstacles(self):
        geometry=dict(bounds=[[-6,6],[-5,5],[0,4.5]],start=[-5,0,1],goal=[5,0,1],
            static_boxes=[[0,0,1,1,7]])
        rows=[dict(state=[-5,0,1],dynamic_cylinders=[[2,1,.5,5,1,0]])]
        project,extent,meta=camera(geometry,rows)
        points=np.array([[x,y,z] for x in [-6,6] for y in [-5,5] for z in [0,7]])
        xy,depth=project(points)
        self.assertTrue((depth>0).all())
        self.assertTrue((abs(xy[:,0])<extent*16/9).all())
        self.assertTrue((abs(xy[:,1])<extent).all())
        self.assertEqual(meta['elevation_deg'],52)
        self.assertEqual(len(cylinder_faces([0,0,1,2])),17)

    def test_terminal_included_and_slowdown(self):
        times=np.arange(31)/30
        normal=frame_indices(times,20,1,0,1,0)
        slow=frame_indices(times,20,.5,0,1,0)
        self.assertEqual(normal[-1],30)
        self.assertGreater(len(slow),len(normal))

    def test_mp4_smoke(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);directory=AnalysisTest().fixture(root)
            output=render(directory,10,root/'test.mp4',fps=5,hold=0,width=640)
            self.assertGreater(output.stat().st_size,1000)
            self.assertTrue(output.with_suffix('.camera.json').exists())
            with self.assertRaises(FileExistsError): render(directory,10,output)

if __name__=='__main__':unittest.main()
