"""Generate or extend a 3D Gaussian scene from posed cameras."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ar_core import (
    GEN_IN_CHANNELS,
    STATIC_PREFIX,
    ARFlashWorld,
    ARWorldSession,
    LatentCodec,
    build_recon_decoder,
    build_timesteps,
    build_transformer,
    build_vae,
    encode_texts,
    interpolate_cameras,
    load_flashworld_ckpt,
    render_frames,
    save_rows,
)
from ar_export import add_export_arguments, check_spz_export, direct_video_frames, export_ply, export_spz
from ar_reference import add_reference_arguments, inference_reference_indices, pack_reference_latents


def positive_int(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return n


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, help="ArgsWorld checkpoint file or directory containing model.ckpt")
    p.add_argument("--model_id", required=True, help="Wan2.2-TI2V-5B-Diffusers model directory or identifier")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--traj_json", type=Path, nargs="+", help="Ordered trajectories; each file begins a new segment")
    p.add_argument("--no_image_cond", action="store_true", help="Run without reference image conditioning")
    p.add_argument("--text", help="Override the prompt for a new session")
    image_group = p.add_mutually_exclusive_group()
    image_group.add_argument("--image", type=Path, help="One reference image")
    image_group.add_argument("--images", type=Path, nargs="+", help="Reference images in camera-slot order")
    add_reference_arguments(p)
    p.add_argument("--height", type=positive_int, help="Output height (default: 480)")
    p.add_argument("--width", type=positive_int, help="Output width (default: 704)")
    p.add_argument("--chunk_size", type=positive_int, help="Maximum views per chunk (default: 24)")
    p.add_argument("--frames_per_view", type=positive_int, default=1,
                   help="Frames per input view in output videos; values above 1 interpolate the rendered camera path")
    p.add_argument("--decode_batch", type=positive_int, default=4, help="Maximum views per Gaussian decoder call")
    p.add_argument("--text_device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--seed", type=int, help="Noise seed (default: 0; resumes restore the saved RNG)")
    p.add_argument("--no_video", action="store_true", help="Skip rendered and direct RGB videos")
    p.add_argument("--save_ply", action="store_true", help="Export final 3D Gaussians as world.ply")
    p.add_argument("--save_session", action="store_true", help="Write session.pt with all continuation tensors")
    p.add_argument("--resume_session", type=Path, help="Extend a previously saved session.pt")
    p.add_argument("--query_json", type=Path, help="Render additional calibrated views without updating the scene")
    add_export_arguments(p)
    args = p.parse_args(argv)
    if args.traj_json is None and not (args.resume_session and args.query_json):
        p.error("Provide --traj_json to generate cameras, or --resume_session with --query_json")
    return args


def crop_transform(source_width, source_height, height, width):
    """Return (left, top, crop width, crop height, x scale, y scale)."""
    if source_width < 1 or source_height < 1:
        raise ValueError("Camera image dimensions must be positive")
    if source_width * height >= source_height * width:
        crop_width, crop_height = max(1, source_height * width // height), source_height
    else:
        crop_width, crop_height = source_width, max(1, source_width * height // width)
    left, top = (source_width - crop_width) // 2, (source_height - crop_height) // 2
    return left, top, crop_width, crop_height, width / crop_width, height / crop_height


def load_reference_images(paths, height, width):
    images, sizes = [], []
    for path in paths:
        with Image.open(path) as source:
            sizes.append(source.size)
            left, top, cw, ch, _, _ = crop_transform(*source.size, height, width)
            image = source.convert("RGB").crop((left, top, left + cw, top + ch))
            image = image.resize((width, height), Image.Resampling.BICUBIC)
            tensor = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 127.5 - 1
            images.append(tensor)
    return torch.stack(images), sizes


def json_cameras(cameras, height, width, reference_sizes=None, default_size=None):
    """Convert camera-to-world OpenGL poses and pixel intrinsics to model cameras."""
    rows = []
    reference_sizes = reference_sizes or {}
    for i, camera in enumerate(cameras):
        quaternion, position = camera["quaternion"], camera["position"]
        if len(quaternion) != 4 or len(position) != 3:
            raise ValueError("Expected a wxyz quaternion and an xyz camera position")
        fx, fy, cx, cy = (float(camera[k]) for k in ("fx", "fy", "cx", "cy"))
        if "width" in camera or "height" in camera:
            size = int(camera["width"]), int(camera["height"])
            if i in reference_sizes and size != reference_sizes[i]:
                raise ValueError("Reference image dimensions must match its calibrated camera")
        else:
            size = reference_sizes.get(i, default_size or (width, height))
        left, top, _, _, sx, sy = crop_transform(*size, height, width)
        pose = [float(v) for v in (*quaternion, *position)]
        intrinsics = [fx * sx / width, fy * sy / height, (cx - left) * sx / width, (cy - top) * sy / height]
        if any(not math.isfinite(v) for v in pose + intrinsics) or fx <= 0 or fy <= 0:
            raise ValueError("Cameras must have finite values and positive focal lengths")
        if math.hypot(*pose[:4]) < 1e-8:
            raise ValueError("Camera quaternion must have nonzero norm")
        rows.append(pose + intrinsics)
    return torch.tensor(rows, dtype=torch.float32)


def load_trajectories(args, continuing=False):
    """Load ordered camera segments, a prompt, and any first-chunk reference images."""
    height, width = args.height, args.width
    paths = args.images or ([args.image] if args.image else None)
    explicit = paths or args.reference_indices is not None or args.num_reference_frames is not None
    use_images = not (args.no_image_cond or continuing)
    if explicit and not use_images:
        raise ValueError("New reference images or slots require a new image-conditioned session")

    if continuing and not args.traj_json:
        return [], "", None, []
    source_paths = [Path(p) for p in args.traj_json]
    data = [json.loads(p.read_text(encoding="utf-8")) for p in source_paths]
    if any(not isinstance(d.get("cameras"), list) or not d["cameras"] for d in data):
        raise ValueError("Every trajectory segment must contain at least one camera")
    prompt = data[0].get("text_prompt", "") or ""
    if not isinstance(prompt, str):
        raise ValueError("text_prompt must be a string")
    indices, images, sizes = [], None, []
    if use_images:
        if not paths:
            embedded = data[0].get("image_prompts")
            if embedded is None and data[0].get("image_prompt"):
                embedded = [data[0]["image_prompt"]]
            if embedded is not None:
                if not isinstance(embedded, list) or not embedded or not all(isinstance(p, str) for p in embedded):
                    raise ValueError("image_prompts must be a nonempty list of paths")
                paths = [source_paths[0].parent / p for p in embedded]
        if paths:
            indices = inference_reference_indices(args, min(args.chunk_size, len(data[0]["cameras"])), len(paths))
            images, sizes = load_reference_images(paths, height, width)
        elif explicit:
            raise ValueError("JSON input requires a reference image for every requested reference slot")

    # Cameras may specify their original raster dimensions individually. For
    # FlashWorld-style inputs without dimensions, use the first image raster
    # for unobserved cameras, or the output raster when there are no images.
    segments = [
        json_cameras(d["cameras"], height, width,
                     reference_sizes=dict(zip(indices, sizes)) if i == 0 else None,
                     default_size=sizes[0] if sizes else None)
        for i, d in enumerate(data)
    ]
    return segments, prompt, images, indices


def write_video_stream(path, batches, fps=8):
    import imageio.v2 as imageio

    with imageio.get_writer(path, fps=fps, quality=8, macro_block_size=1) as writer:
        for batch in batches:
            for frame in batch:
                writer.append_data(frame)


def save_query_views(path, session, query, device):
    """Render a query trajectory against the final geometry without changing history."""
    query = json_cameras(query, session.height, session.width).to(device)
    query = session.to_global(query)
    gaussians = session.world.all_gaussians()
    path.mkdir(exist_ok=True)
    offset = 0
    for batch in query.split(8):
        frames = render_frames(session.model, gaussians, batch, session.height, session.width)
        for frame in frames:
            Image.fromarray(frame).save(path / f"{offset:06d}.png")
            offset += 1


def load_argsworld_weights(path):
    """Load the student and decoder from a joint ArgsWorld inference checkpoint."""
    generator, decoder = load_flashworld_ckpt(path, weights_only=True)
    patch_weight = generator.get("patch_embedding.weight")
    if not torch.is_tensor(patch_weight) or patch_weight.ndim != 5 or patch_weight.shape[1] != GEN_IN_CHANNELS:
        raise ValueError(
            "Expected an ArgsWorld student checkpoint with a 1174-channel input. "
            "Supply the trained student rather than the teacher or the base Wan model."
        )
    if not decoder:
        raise ValueError("The inference checkpoint must contain a Gaussian decoder")
    return generator, decoder


def main(argv=None):
    args = parse_args(argv)
    check_spz_export(args)
    if not torch.cuda.is_available():
        raise RuntimeError("Full ArgsWorld inference requires CUDA")
    device = torch.device("cuda")
    state = None
    if args.resume_session is not None:
        if args.text is not None:
            raise ValueError("A resumed session retains its original text conditioning")
        state = torch.load(args.resume_session, map_location="cpu", weights_only=True, mmap=True)
        for key in ("height", "width", "chunk_size"):
            if getattr(args, key) not in (None, state[key]):
                raise ValueError(f"--{key} must match the saved session")
            setattr(args, key, state[key])
    else:
        args.height = args.height or 480
        args.width = args.width or 704
        args.chunk_size = args.chunk_size or 24
    height, width = args.height, args.width
    if height % 32 or width % 32:
        raise ValueError("Image height and width must be multiples of 32")
    segments, text, images, slots = load_trajectories(args, continuing=state is not None)
    query_cameras = None
    if args.query_json is not None:
        query_cameras = json.loads(args.query_json.read_text(encoding="utf-8")).get("cameras")
        if not isinstance(query_cameras, list) or not query_cameras:
            raise ValueError("--query_json requires a nonempty cameras list")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gen_sd, recon_sd = load_argsworld_weights(args.ckpt)
    if state is None:
        prompt = STATIC_PREFIX + (text if args.text is None else args.text)
        text_emb = encode_texts([prompt], args.model_id, torch.device(args.text_device))[prompt][None].to(device)

    generator = build_transformer(args.model_id, GEN_IN_CHANNELS, gen_sd, device, dtype=torch.bfloat16)
    generator.eval().requires_grad_(False)
    vae = build_vae(args.model_id)
    codec = LatentCodec(vae).to(device)
    recon = build_recon_decoder(vae, recon_sd, network_checkpointing=False, render_checkpointing=False)
    recon.to(device).eval().requires_grad_(False)
    del gen_sd, recon_sd
    ar = ARFlashWorld(generator, recon, codec, build_timesteps(args.model_id).to(device),
                      vae_micro_batch=4, recon_micro_batch_nograd=args.decode_batch)

    if state is not None:
        session = ARWorldSession.from_state_dict(ar, state, device)
        if args.seed is None:
            torch.set_rng_state(state["cpu_rng_state"])
            torch.cuda.set_rng_state(state["cuda_rng_state"], device)
        else:
            torch.manual_seed(args.seed)
        del state
    else:
        torch.manual_seed(0 if args.seed is None else args.seed)
        image_latent = None
        if images is not None:
            # Match the evaluation pipeline: encode the reference in full
            # precision before entering autocast for autoregressive sampling.
            with torch.no_grad():
                refs = codec.encode(images[None].to(device), micro_batch=ar.vae_micro_batch)
                image_latent = pack_reference_latents(refs, slots)
            save_rows(str(args.out_dir / "reference_views.jpg"), [images], max_cols=8, downsample=1)
        (args.out_dir / "reference_inputs.json").write_text(
            json.dumps({"reference_indices": slots}, indent=2) + "\n", encoding="utf-8"
        )
        session = ARWorldSession(ar, text_emb, height, width, args.chunk_size, image_latent)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, cameras in enumerate(segments, start=1):
            session.extend(cameras.to(device))
            print(f"Segment {i}: {len(cameras)} new views, {session.world.num_views} total views", flush=True)

    gaussians = session.world.all_gaussians()
    cameras = session.world.all_cams()
    if args.save_session:
        # Save the sampling state before any optional visualization or export.
        state = session.state_dict()
        state["cpu_rng_state"] = torch.get_rng_state()
        state["cuda_rng_state"] = torch.cuda.get_rng_state(device)
        torch.save(state, args.out_dir / "session.pt")
        del state
    if not args.no_video:
        path = interpolate_cameras(cameras, args.frames_per_view)
        write_video_stream(args.out_dir / "final_world.mp4",
                           (render_frames(ar, gaussians, batch, height, width) for batch in path.split(8)))
        history = session.world.latents[0][None]
        write_video_stream(args.out_dir / "direct_rgb.mp4",
                           (direct_video_frames(codec, batch, args.frames_per_view)
                            for batch in history.split(args.chunk_size, dim=2)))
    if args.save_ply:
        export_ply(args.out_dir / "world.ply", gaussians)
    if args.save_spz:
        export_spz(args.out_dir / "world.spz", gaussians, args.spz_opacity_threshold)
    if query_cameras is not None:
        save_query_views(args.out_dir / "query_views", session, query_cameras, device)
    torch.save(cameras.cpu(), args.out_dir / "session_cameras.pt")


if __name__ == "__main__":
    main()
