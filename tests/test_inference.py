"""Exercise paper inference semantics without weights, CUDA, or external data."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from ar_core import ARFlashWorld, ARWorldSession, FEAT_DIM, GEN_IN_CHANNELS, LATENT_DIM, build_timesteps
from ar_reference import pack_reference_latents
from infer_ar import load_argsworld_weights, load_trajectories, parse_args, save_query_views


def trajectory(count):
    cameras = torch.zeros(count, 11)
    cameras[:, 0] = 1
    cameras[:, 4] = torch.arange(count) + 3
    cameras[:, 7:] = torch.tensor([0.8, 0.8, 0.5, 0.5])
    return cameras


class TinyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, hidden_states, timestep, **kwargs):
        self.inputs.append((hidden_states.detach().clone(), timestep.clone()))
        x = hidden_states[:, :LATENT_DIM]
        value = x.mean() + timestep.max() / 1000
        features = value.expand(1, FEAT_DIM, x.shape[2], *x.shape[-2:])
        return (torch.cat([x * 0.1, features], dim=1),)


class TinyCodec:
    def encode(self, images, **kwargs):
        value = images.mean(dim=(2, 3, 4))
        return value[:, None, :, None, None].expand(1, LATENT_DIM, value.shape[1], 2, 2)

    def unscale(self, latents):
        return latents


class TinyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoded_views = []
        self.rendered_views = []

    def forward(self, features, latents, cameras):
        assert not latents.requires_grad, "Decoder inputs must be detached"
        self.decoded_views.append(cameras.shape[1])
        value = features.mean(dim=(1, 2, 3, 4))[None, :, None]
        return torch.cat([cameras[..., 4:7], value.expand(1, cameras.shape[1], 35)], dim=-1)

    def render(self, gaussians, cameras, height, width, bg_mode):
        self.rendered_views.append((len(gaussians[0]), cameras.shape[1], bg_mode))
        value = gaussians[0][:, 3].mean()
        return value.expand(1, cameras.shape[1], 3, height, width), None


def make_model():
    return ARFlashWorld(TinyStudent(), TinyDecoder(), TinyCodec(),
                        build_timesteps(None), recon_micro_batch_nograd=None)


class InferenceTests(unittest.TestCase):
    def test_first_segment_sets_normalization_for_later_segments(self):
        session = ARWorldSession(make_model(), None, 32, 32, 2)
        cameras = trajectory(6)
        first = session.to_global(cameras[:4])
        # The first segment has four views even though chunks have two.
        torch.testing.assert_close(session.T_norm, torch.tensor([[[3.0]]]))
        torch.testing.assert_close(first[:, 4], torch.tensor([0., 1., 2., 3.]) / 3.01)
        later = session.to_global(cameras[4:])
        torch.testing.assert_close(session.T_norm, torch.tensor([[[3.0]]]))
        torch.testing.assert_close(later[:, 4], torch.tensor([4., 5.]) / 3.01)

    def test_schedule_reference_slots_and_complete_scene_rebuild(self):
        torch.manual_seed(7)
        model = make_model()
        refs = torch.ones(1, LATENT_DIM, 2, 2, 2)
        refs[:, :, 1] = 2
        session = ARWorldSession(model, None, 32, 32, 4, pack_reference_latents(refs, [3, 0]))
        cams = trajectory(7)
        session.extend(cams[:4])
        old_geometry = session.world.all_gaussians().clone()
        old_cams, old_history, old_features = session.world.history()
        session.extend(cams[4:])
        history_cams, history, features = session.world.history()
        hidden, times = model.generator.inputs[0]
        torch.testing.assert_close(hidden[:, 54:102, [3, 0]], refs)
        self.assertEqual(hidden[:, 54:102, [1, 2]].count_nonzero(), 0)
        self.assertEqual(times.tolist(), [1000])
        torch.testing.assert_close(
            torch.stack([t.max() for _, t in model.generator.inputs[:4]]) / 1000,
            torch.tensor([1.0, 0.9, 0.75, 0.5]),
        )
        hidden, times = model.generator.inputs[4]
        torch.testing.assert_close(times[0], torch.tensor([0.] * 4 + [1000.] * 3))
        torch.testing.assert_close(hidden[0, :48, :4], old_history)
        torch.testing.assert_close(hidden[0, 54:102, :4], old_history)
        torch.testing.assert_close(hidden[0, 102:1126, :4], old_features)
        torch.testing.assert_close(hidden[0, 1126:, :4], old_history)
        self.assertEqual(hidden[0, 54:102, 4:].count_nonzero(), 0)
        torch.testing.assert_close(history_cams[:4], old_cams)
        torch.testing.assert_close(history[:, :4], old_history, rtol=0, atol=0)
        self.assertFalse(torch.equal(features[:, :4], old_features))
        self.assertFalse(torch.equal(session.world.all_gaussians()[:4], old_geometry))
        self.assertEqual(model.recon.decoded_views, [4] * 4 + [7] * 4)
        self.assertEqual(model.recon.rendered_views, [(4, 4, "white")] * 3 + [(7, 3, "white")] * 3)
        torch.testing.assert_close(history_cams[:, 4], torch.arange(7) / 3.01)

    def test_saved_session_matches_uninterrupted_generation(self):
        torch.manual_seed(9)
        cams = trajectory(6)
        session = ARWorldSession(make_model(), None, 32, 32, 4)
        session.extend(cams[:4])
        rng = torch.get_rng_state()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.pt"
            torch.save(session.state_dict(), path)
            saved = torch.load(path, weights_only=True)
            resumed = ARWorldSession.from_state_dict(make_model(), saved, torch.device("cpu"))
        session.extend(cams[4:])
        torch.set_rng_state(rng)
        resumed.extend(cams[4:])
        for original, restored in zip(session.world.history(), resumed.world.history()):
            torch.testing.assert_close(original, restored, rtol=0, atol=0)
        torch.testing.assert_close(session.world.all_gaussians(), resumed.world.all_gaussians(), rtol=0, atol=0)

    def test_json_cameras_follow_reference_order_and_crop_intrinsics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (48, 32), 0).save(root / "ref_a.png")
            Image.new("RGB", (32, 48), (255, 255, 255)).save(root / "ref_b.png")
            cameras = [dict(quaternion=[1, 0, 0, 0], position=[i, 0, 0],
                            fx=32, fy=32, cx=24, cy=16) for i in range(4)]
            path = root / "segment.json"
            path.write_text(json.dumps(dict(cameras=cameras, image_prompts=["ref_a.png", "ref_b.png"])),
                            encoding="utf-8")
            args = parse_args([
                "--model_id", "base", "--ckpt", "student.ckpt", "--out_dir", str(root),
                "--traj_json", str(path), "--reference_indices", "3", "0",
                "--height", "32", "--width", "32", "--chunk_size", "4",
            ])
            segments, _, images, slots = load_trajectories(args)
            self.assertFalse(args.save_spz)
            self.assertEqual(slots, [3, 0])
            torch.testing.assert_close(images[:, 0, 0, 0], torch.tensor([-1., 1.]))
            torch.testing.assert_close(segments[0][slots, 9:], torch.tensor([[.5, .5], [.75, .25]]))
            with self.assertRaises(ValueError):
                load_trajectories(args, continuing=True)

    def test_inference_weights_accept_packaged_and_prefixed_states(self):
        student = {"patch_embedding.weight": torch.zeros(1, GEN_IN_CHANNELS, 1, 2, 2)}
        decoder = {"parameter": torch.ones(1)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "model.ckpt"
            for checkpoint in [
                {"transformer": student, "recon_decoder": decoder},
                {"state_dict": {"transformer.patch_embedding.weight": student["patch_embedding.weight"],
                                "recon_decoder.parameter": decoder["parameter"]}},
            ]:
                torch.save(checkpoint, path)
                actual_student, actual_decoder = load_argsworld_weights(root)
                torch.testing.assert_close(actual_student, student)
                torch.testing.assert_close(actual_decoder, decoder)
            teacher = {"patch_embedding.weight": torch.zeros(1, 102, 1, 2, 2)}
            torch.save({"transformer": teacher, "recon_decoder": decoder}, path)
            with self.assertRaisesRegex(ValueError, "1174-channel"):
                load_argsworld_weights(path)
            torch.save({"transformer": student}, path)
            with self.assertRaisesRegex(ValueError, "Gaussian decoder"):
                load_argsworld_weights(path)

    def test_query_renders_without_extending_the_world(self):
        session = ARWorldSession(make_model(), None, 32, 32, 2)
        session.extend(trajectory(2))
        before = session.world.all_cams().clone()
        query = [dict(quaternion=[1, 0, 0, 0], position=[5, 0, 0], fx=22, fy=22, cx=16, cy=16)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query_views"
            save_query_views(path, session, query, torch.device("cpu"))
            with Image.open(path / "000000.png") as frame:
                self.assertEqual(frame.size, (32, 32))
        torch.testing.assert_close(session.world.all_cams(), before)


if __name__ == "__main__":
    unittest.main()
