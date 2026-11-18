import json
import tempfile
import unittest
from pathlib import Path

from infer import build_inference_args, parse_args


class SimpleInferenceTests(unittest.TestCase):
    def test_modes_and_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'trajectory.json').write_text(json.dumps({'cameras': [{}] * 24}), encoding='utf-8')
            (root / 'a.png').touch()
            (root / 'b.png').touch()
            common = ['--ckpt', 'model.ckpt', '--model_id', 'wan', '--input_dir', str(root),
                      '--output_dir', str(root / 'output')]
            for mode in ('reconstruction', 'generation'):
                args = parse_args(['--mode', mode, *common, '--images', 'a.png', 'b.png',
                                   '--reference_indices', '0', '2'])
                forwarded = build_inference_args(args)
                self.assertEqual(forwarded[forwarded.index('--images') + 1], str(root / 'a.png'))
                self.assertIn(str(root / 'trajectory.json'), forwarded)
                self.assertEqual(forwarded[-2:], ['0', '2'])
            args = parse_args(['--mode', 'generation', *common])
            self.assertNotIn('--images', build_inference_args(args))
            for images in ([], ['--images', 'a.png']):
                args = parse_args(['--mode', 'reconstruction', *common, *images])
                with self.assertRaisesRegex(ValueError, 'at least two reference images'):
                    build_inference_args(args)
            args = parse_args(['--mode', 'generation', *common, '--images', 'a.png'])
            self.assertIn(str(root / 'a.png'), build_inference_args(args))
            (root / 'trajectory.json').write_text(json.dumps({
                'cameras': [{}] * 9, 'image_prompts': ['a.png', 'b.png']
            }), encoding='utf-8')
            args = parse_args(['--mode', 'reconstruction', *common])
            self.assertIn(str(root / 'trajectory.json'), build_inference_args(args))
            # Explicit images override embedded paths, including the reference count.
            args = parse_args(['--mode', 'reconstruction', *common, '--images', 'a.png'])
            with self.assertRaisesRegex(ValueError, 'at least two reference images'):
                build_inference_args(args)
            (root / 'trajectory.json').write_text(json.dumps({
                'cameras': [{}] * 9, 'image_prompt': 'a.png'
            }), encoding='utf-8')
            args = parse_args(['--mode', 'reconstruction', *common])
            with self.assertRaisesRegex(ValueError, 'at least two reference images'):
                build_inference_args(args)
            (root / 'trajectory.json').write_text(json.dumps({'cameras': [{}]}), encoding='utf-8')
            args = parse_args(['--mode', 'extrapolation', *common])
            with self.assertRaisesRegex(ValueError, 'more than 24 cameras'):
                build_inference_args(args)
            (root / 'second.json').write_text(json.dumps({'cameras': [{}] * 24}), encoding='utf-8')
            args = parse_args(['--mode', 'extrapolation', *common, '--trajectory', 'trajectory.json', 'second.json'])
            self.assertIn(str(root / 'second.json'), build_inference_args(args))
            args = parse_args(['--mode', 'generation', *common, '--trajectory', 'trajectory.json', 'second.json'])
            with self.assertRaisesRegex(ValueError, 'one trajectory'):
                build_inference_args(args)
            (root / 'trajectory.json').write_text(json.dumps({'cameras': [{}] * 25}), encoding='utf-8')
            args = parse_args(['--mode', 'generation', *common])
            with self.assertRaisesRegex(ValueError, 'at most 24 cameras'):
                build_inference_args(args)


if __name__ == '__main__':
    unittest.main()
