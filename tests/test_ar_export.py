"""Real SPZ serialization and direct-video/session regression tests."""
from importlib.util import find_spec
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import sys

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ar_export import export_spz, export_ply, direct_video_frames
from ar_core import ARWorldSession, write_video


HAS_SPZ = find_spec("spz") is not None


class ExportTests(unittest.TestCase):
    @unittest.skipUnless(HAS_SPZ, "Optional SPZ bindings are not installed")
    def test_spz_roundtrip_preserves_coordinates_sh_and_rotation(self):
        import spz
        g = torch.zeros(3, 38)
        g[:, :3] = torch.tensor([[1., 2., 3.], [-2., .5, -1.], [9., 9., 9.]])
        g[:, 3] = torch.tensor([.7, .9, .00001])
        g[:, 4:7] = torch.tensor([.1, .3, .7])
        g[:, 7:11] = torch.tensor([[1., 0., 0., 0.], [1., 2., 3., 4.], [1., 0., 0., 0.]])
        # Use representable SH values: SPZ quantizes higher bands more coarsely.
        g[:, 11:] = ((torch.arange(27) % 7) - 3) * .125
        original = g.clone()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'world.spz'
            self.assertEqual(export_spz(path, g), 2)
            cloud = spz.load_spz(str(path))
            self.assertEqual(cloud.sh_degree, 2)
            np.testing.assert_allclose(np.asarray(cloud.positions).reshape(-1, 3), g[:2, :3], atol=.001)
            np.testing.assert_allclose(np.asarray(cloud.scales).reshape(-1, 3), g[:2, 4:7].log(), atol=.04)
            np.testing.assert_allclose(np.asarray(cloud.colors).reshape(-1, 3), g[:2, 11:14], atol=.02)
            np.testing.assert_allclose(np.asarray(cloud.sh).reshape(2, -1), g[:2, 14:], atol=.05)
            np.testing.assert_allclose(np.asarray(cloud.alphas), torch.logit(g[:2, 3]), atol=.05)
            # SPZ exposes xyzw; gsplat/model rotations are wxyz. Check the
            # physical rotation, independently of the exporter's permutation.
            from utils import quaternion_to_matrix
            actual = np.asarray(cloud.rotations).reshape(-1, 4)
            decoded = torch.from_numpy(actual[:, [3, 0, 1, 2]].copy())
            torch.testing.assert_close(quaternion_to_matrix(decoded),
                                       quaternion_to_matrix(g[:2, 7:11]), atol=.01, rtol=0)
        torch.testing.assert_close(g, original)

    def test_ply_uses_standard_sh_layout_and_unquantized_values(self):
        from plyfile import PlyData
        g = torch.zeros(2, 38)
        g[:, :3] = torch.tensor([[1.25, -2., 3.], [0., 0.5, -1.]])
        g[:, 3] = torch.tensor([.25, .75])
        g[:, 4:7] = torch.tensor([[.01, .02, .03], [.2, .3, .4]])
        g[:, 7:11] = torch.tensor([[1., 0., 0., 0.], [.5, .5, .5, .5]])
        g[:, 11:] = torch.arange(27).float() * .01
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'world.ply'
            self.assertEqual(export_ply(path, g, chunk_size=1), 2)
            ply = PlyData.read(path)['vertex']
            names = ('x', 'y', 'z', *(f'f_dc_{i}' for i in range(3)),
                     *(f'f_rest_{i}' for i in range(24)),
                     'opacity', 'scale_0', 'scale_1', 'scale_2',
                     'rot_0', 'rot_1', 'rot_2', 'rot_3')
            self.assertEqual(ply.data.dtype.names, names)
            self.assertAlmostEqual(float(ply['f_rest_23'][1]), float(g[1, 37]), places=6)
            self.assertAlmostEqual(float(ply['opacity'][0]), float(torch.logit(g[0, 3])), places=6)
            self.assertAlmostEqual(float(ply['scale_0'][0]), float(g[0, 4].log()), places=5)
            self.assertAlmostEqual(float(ply['rot_0'][0]), 1.)
            if not HAS_SPZ:
                return
            # Use SPZ's independent standard PLY parser: it restores the
            # coefficient-major SH and xyzw rotations used by GaussianCloud.
            import spz
            cloud = spz.load_splat_from_ply(str(path))
            self.assertEqual(cloud.sh_degree, 2)
            np.testing.assert_allclose(np.asarray(cloud.colors).reshape(2, 3), g[:, 11:14], atol=1e-6)
            np.testing.assert_allclose(np.asarray(cloud.sh).reshape(2, 24), g[:, 14:], atol=1e-6)
            np.testing.assert_allclose(np.asarray(cloud.rotations).reshape(2, 4),
                                       g[:, [8, 9, 10, 7]], atol=1e-6)

    @unittest.skipUnless(HAS_SPZ, "Optional SPZ bindings are not installed")
    def test_ply_degree_zero_and_all_supported_sh_degrees(self):
        import spz
        with tempfile.TemporaryDirectory() as tmp:
            for degree in range(4):
                with self.subTest(degree=degree):
                    channels = 3 * (degree + 1) ** 2
                    g = torch.zeros(2, 11 + channels)
                    g[:, 3] = .5
                    g[:, 4:7] = .1
                    g[:, 7] = 1
                    g[:, 11:] = torch.arange(channels) * .01
                    path = Path(tmp) / 'world.ply'
                    export_ply(path, g, chunk_size=1)
                    cloud = spz.load_splat_from_ply(str(path))
                    self.assertEqual(cloud.sh_degree, degree)
                    np.testing.assert_allclose(np.asarray(cloud.sh).reshape(2, channels - 3),
                                               g[:, 14:], atol=1e-6)

    @unittest.skipUnless(HAS_SPZ, "Optional SPZ bindings are not installed")
    def test_spz_rejects_empty_invalid_scale_and_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            for g in [torch.zeros(0, 38), torch.zeros(1, 37), torch.zeros(1, 38)]:
                with self.assertRaises(ValueError):
                    export_spz(Path(tmp)/'bad.spz', g, opacity_threshold=0)

    def test_direct_video_decodes_latents_and_holds_each_view(self):
        latents = torch.randn(1, 48, 3, 2, 2, dtype=torch.bfloat16)
        rgb = torch.tensor([-1., 0., 1.])[None, :, None, None, None].expand(1, 3, 3, 16, 16)
        codec = SimpleNamespace(decode=Mock(return_value=rgb))
        frames = direct_video_frames(codec, latents, 2)
        self.assertEqual(codec.decode.call_args.args[0].dtype, torch.float32)
        torch.testing.assert_close(codec.decode.call_args.args[0], latents.float())
        self.assertEqual(frames.shape, (6, 16, 16, 3))
        self.assertEqual(frames[:, 0, 0, 0].tolist(), [0, 0, 128, 128, 255, 255])
        # Exercise the actual video writer and reader, not only an in-memory array.
        import imageio.v2 as imageio
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/'dit_video.mp4')
            write_video(path, frames)
            with imageio.get_reader(path) as reader:
                decoded = list(reader.iter_data())
            self.assertEqual(len(decoded), 6)
            self.assertEqual(decoded[0].shape, (16, 16, 3))

    def test_session_returns_all_chunk_outputs_only_when_requested(self):
        model = SimpleNamespace(rollout=Mock(return_value=(None, [{'x0_2d': 1}, {'x0_2d': 2}])))
        session = ARWorldSession(model, None, 32, 32, 8)
        cameras = torch.randn(12, 11)
        with patch.object(session, 'to_global', return_value=cameras):
            self.assertIs(session.extend(cameras), cameras)
            cams, outputs = session.extend(cameras, return_outputs=True)
        self.assertIs(cams, cameras)
        self.assertEqual([o['x0_2d'] for o in outputs], [1, 2])
