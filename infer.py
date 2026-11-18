"""Simple command-line entry point for ArgsWorld inference."""
import argparse
import json
from pathlib import Path

CHUNK_SIZE = 24

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('reconstruction', 'generation', 'extrapolation'), required=True)
    parser.add_argument('--ckpt', required=True, help='Trained ArgsWorld checkpoint')
    parser.add_argument('--model_id', required=True, help='Wan2.2-TI2V-5B-Diffusers directory')
    parser.add_argument('--input_dir', type=Path, required=True, help='Directory containing input files')
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--trajectory', nargs='+', type=Path, default=[Path('trajectory.json')],
                        help='Ordered JSON trajectory files, relative to input_dir')
    parser.add_argument('--images', nargs='+', type=Path,
                        help='One or more reference images, relative to input_dir')
    parser.add_argument('--reference_indices', nargs='+', type=int,
                        help='Zero-based camera slots for the images in the first chunk')
    return parser.parse_args(argv)


def build_inference_args(args):
    def input_path(path):
        return path if path.is_absolute() else args.input_dir / path

    trajectories = [input_path(path) for path in args.trajectory]
    for path in trajectories:
        if not path.is_file():
            raise ValueError(f'Trajectory file does not exist: {path}')
    data = [json.loads(path.read_text(encoding='utf-8')) for path in trajectories]
    if any(not isinstance(d, dict) or not isinstance(d.get('cameras'), list) or not d['cameras'] for d in data):
        raise ValueError('Each trajectory must contain a nonempty cameras list')
    embedded = []
    if not args.images:
        embedded = data[0].get('image_prompts')
        if embedded is None:
            embedded = [data[0]['image_prompt']] if data[0].get('image_prompt') else []
        if not isinstance(embedded, list) or not all(isinstance(p, str) and p for p in embedded):
            raise ValueError('image_prompts must be a list of image paths')
    if args.mode == 'reconstruction' and len(args.images or embedded) < 2:
        raise ValueError('Reconstruction requires at least two reference images of the same scene')
    if args.mode == 'generation' and (len(trajectories) != 1 or len(data[0]['cameras']) > CHUNK_SIZE):
        raise ValueError('Generation uses one trajectory with at most 24 cameras (one chunk)')
    if args.mode == 'extrapolation' and sum(len(d['cameras']) for d in data) <= CHUNK_SIZE:
        raise ValueError('Extrapolation requires more than 24 cameras')
    if args.reference_indices and not (args.images or embedded):
        raise ValueError('Reference indices require reference images')
    result = ['--ckpt', args.ckpt, '--model_id', args.model_id, '--out_dir', str(args.output_dir),
              '--save_ply', '--chunk_size', str(CHUNK_SIZE), '--traj_json', *map(str, trajectories)]
    if args.images:
        images = [input_path(path) for path in args.images]
        for path in images:
            if not path.is_file():
                raise ValueError(f'Image file does not exist: {path}')
        result.extend(['--images', *map(str, images)])
    if args.reference_indices:
        result.extend(['--reference_indices', *map(str, args.reference_indices)])
    return result


def main(argv=None):
    args = parse_args(argv)
    try:
        inference_args = build_inference_args(args)
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    from infer_ar import main as run_inference
    run_inference(inference_args)


if __name__ == '__main__':
    main()
