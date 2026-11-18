"""ArgsWorld inference, autoregressive rollout, and Gaussian reconstruction utilities.

At each denoising step, the DiT sees the complete clean RGB prefix and the
current noisy views. The decoder rebuilds all Gaussians from the refreshed
features, the detached clean RGB predictions, and the prefix cameras. The
next step uses rendering feedback; the next chunk uses the final direct RGB
latents and refreshed features.
"""
from typing import List, Sequence

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from torch.utils.checkpoint import checkpoint

from models import AutoencoderKLWan, WanTransformer3DModel, WANDecoderPixelAligned3DGSReconstructionModel
from models.autoencoder_kl_wan import WanCausalConv3d
from utils import create_raymaps, normalize_cameras

LATENT_DIM = 48
FEAT_DIM = 1024
RAYMAP_DIM = 6
SPATIAL_DOWN = 16
PATCH_HW = 2
OUT_CHANNELS = LATENT_DIM + FEAT_DIM
# noisy latent | raymap | condition latent | feedback feats | feedback latents
GEN_IN_CHANNELS = LATENT_DIM + RAYMAP_DIM + LATENT_DIM + FEAT_DIM + LATENT_DIM
DEFAULT_DENOISING_STEPS = (0, 250, 500, 750)
STATIC_PREFIX = "[Static] "


# ----------------------------------------------------------------------------------------------
# model construction
# ----------------------------------------------------------------------------------------------

def load_flashworld_ckpt(path, with_config=False, *, weights_only=False):
    """Returns (transformer_sd, recon_decoder_sd). Accepts the FlashWorld release format
    {'transformer', 'recon_decoder'} and Lightning checkpoints with prefixed keys."""
    from pathlib import Path
    if Path(path).is_dir():
        path = Path(path)/"model.ckpt"
    ckpt = torch.load(path, map_location="cpu", weights_only=weights_only, mmap=True)
    if "transformer" in ckpt:
        result = (ckpt["transformer"], ckpt.get("recon_decoder"))
    else:
        sd = ckpt.get("state_dict", ckpt)
        tr = {k[len("transformer."):]: v for k, v in sd.items() if k.startswith("transformer.")}
        rc = {k[len("recon_decoder."):]: v for k, v in sd.items() if k.startswith("recon_decoder.")}
        result = (tr, rc or None)
    return (*result, ckpt.get("config", {})) if with_config else result


def adapt_transformer_state_dict(sd, in_channels, num_layers=None):
    """Slices / zero-pads the patch embedding to `in_channels` (zero feedback channels contribute
    nothing, so slicing 1174 -> 102 gives exactly the generator's no-feedback behaviour)."""
    out = {}
    for k, v in sd.items():
        if num_layers is not None and k.startswith("blocks.") and int(k.split(".")[1]) >= num_layers:
            continue
        if k == "patch_embedding.weight" and v.shape[1] != in_channels:
            if v.shape[1] > in_channels:
                v = v[:, :in_channels]
            else:
                v = F.pad(v, (0, 0, 0, 0, 0, 0, 0, in_channels - v.shape[1]))
        out[k] = v
    return out


def build_transformer(model_id, in_channels, state_dict, device, num_layers=None, dtype=torch.float32):
    config = dict(WanTransformer3DModel.load_config(model_id, subfolder="transformer"))
    config.update(in_channels=in_channels, out_channels=OUT_CHANNELS)
    if num_layers is not None:
        config["num_layers"] = num_layers
    with init_empty_weights(include_buffers=False):
        model = WanTransformer3DModel.from_config(config)
    sd = adapt_transformer_state_dict(state_dict, in_channels, num_layers)
    sd = {k: v.to(device=device, dtype=dtype, copy=True).contiguous() for k, v in sd.items()}
    model.load_state_dict(sd, strict=True, assign=True)
    return model.to(device)


def build_vae(model_id):
    """Wan2.2 VAE with the causal time padding removed (FlashWorld encodes every view as an image)."""
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    with torch.no_grad():
        for m in vae.modules():
            if isinstance(m, WanCausalConv3d):
                time_pad = m._padding[4]
                m.padding = (0, m._padding[2], m._padding[0])
                m._padding = (0, 0, 0, 0, 0, 0)
                m.weight = nn.Parameter(m.weight[:, :, time_pad:].clone())
    return vae.eval().requires_grad_(False)


def build_recon_decoder(vae, state_dict, network_checkpointing=True, render_checkpointing=True):
    recon = WANDecoderPixelAligned3DGSReconstructionModel(
        vae, FEAT_DIM, use_render_checkpointing=render_checkpointing, use_network_checkpointing=network_checkpointing)
    if state_dict is not None:
        recon.load_state_dict(state_dict, strict=True)
    return recon


def build_timesteps(model_id, shift=3.0):
    from diffusers import FlowMatchEulerDiscreteScheduler
    # Appendix A uses the 1,000-entry flow-matching schedule, independent of
    # the solver configuration distributed with the base Wan checkpoint.
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift)
    return scheduler.timesteps.clone().float()  # sigma = t / 1000


@torch.no_grad()
def encode_texts(texts: Sequence[str], model_id, device, batch_size=8, max_len=512, sink=None):
    """UMT5 embeddings padded to `max_len` exactly like FlashWorld. Returns {text: [max_len, 4096] cpu}."""
    from transformers import AutoTokenizer, UMT5EncoderModel
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float32)
    encoder = encoder.eval().requires_grad_(False).to(device)
    texts = list(dict.fromkeys(texts))
    cache = {}
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tok = tokenizer(batch, padding="max_length", max_length=max_len, truncation=True,
                        add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
        ids, mask = tok.input_ids.to(device), tok.attention_mask.to(device)
        emb = encoder(ids, mask).last_hidden_state.float()
        for text, e, n in zip(batch, emb, mask.sum(1).tolist()):
            e = e.clone()
            e[n:] = 0
            if sink is None:
                cache[text] = e.cpu()
            else:
                sink(text, e[:n].to(dtype=torch.bfloat16, device="cpu"))
    del encoder
    torch.cuda.empty_cache()
    return cache


class LatentCodec(nn.Module):
    """FlashWorld latent space: per-view VAE posterior samples, normalised by the Wan2.2 stats."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        self.register_buffer("mean", torch.tensor(vae.config.latents_mean).float().view(1, -1, 1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(vae.config.latents_std).float().view(1, -1, 1, 1, 1), persistent=False)

    def scale(self, z):
        return (z - self.mean) / self.std

    def unscale(self, z):
        return z * self.std + self.mean

    def encode(self, images, use_checkpoint=False, micro_batch=None):
        """images [B,T,3,H,W] in [-1,1] -> scaled latents [B,48,T,h,w]."""
        B = images.shape[0]
        x = einops.rearrange(images, "B T C H W -> (B T) C H W").unsqueeze(2)
        moments = []
        for xc in (x.split(micro_batch) if micro_batch else [x]):
            if use_checkpoint and torch.is_grad_enabled():
                moments.append(checkpoint(self.vae._encode, xc, use_reentrant=False))
            else:
                moments.append(self.vae._encode(xc))
        z = DiagonalGaussianDistribution(torch.cat(moments)).sample()
        return einops.rearrange(self.scale(z).squeeze(2), "(B T) C H W -> B C T H W", B=B)

    @torch.no_grad()
    def decode(self, latents, micro_batch=4):
        """scaled latents [B,48,T,h,w] -> images [B,T,3,H,W] in [-1,1] (visualisation only)."""
        B = latents.shape[0]
        z = einops.rearrange(self.unscale(latents), "B C T H W -> (B T) C H W").unsqueeze(2)
        imgs = torch.cat([self.vae.decode(zc, return_dict=False)[0] for zc in z.split(micro_batch)])
        return einops.rearrange(imgs.squeeze(2), "(B T) C H W -> B T C H W", B=B).clamp(-1, 1)


# ----------------------------------------------------------------------------------------------
# cameras
# ----------------------------------------------------------------------------------------------

def raymaps_for(cams, h, w):
    """Encode each camera ray by its direction and closest point to the origin."""
    return einops.rearrange(create_raymaps(cams[None].float(), h, w), "B T H W C -> B C T H W")


def local_normalize(cams):
    """Translate / scale a camera window so its first camera is at the origin and the farthest one at
    distance ~1, i.e. FlashWorld's training normalisation without the rotation part. Without rotation
    the Gaussians map back by a similarity transform that leaves rotations and SH untouched."""
    origin = cams[0, 4:7].clone()
    rel = cams[:, 4:7] - origin
    scale = rel.norm(dim=-1).max() + 1e-2
    out = cams.clone()
    out[:, 4:7] = rel / scale
    return out, origin, scale


def gaussians_to_global(gs, origin, scale):
    xyz, opacity, scales, rest = gs.split([3, 1, 3, gs.shape[-1] - 7], dim=-1)
    return torch.cat([xyz * scale + origin, opacity, scales * scale, rest], dim=-1)


class ARGaussianWorld:
    """Latest complete 3DGS reconstruction of a single scene."""

    def __init__(self):
        self.gaussians: List[torch.Tensor] = []  # [N_i, 38]
        self.cams: List[torch.Tensor] = []       # [K_i, 11] global frame
        self.latents: List[torch.Tensor] = []    # [48, K_i, h, w]   history image latents (fully denoised DiT x0)
        self.feats: List[torch.Tensor] = []      # [1024, K_i, h, w] history DiT features (last sampling step)

    def __len__(self):
        return len(self.gaussians)

    @property
    def num_views(self):
        return sum(c.shape[0] for c in self.cams)

    def replace_all(self, gaussians, cams, latents=None, feats=None):
        """Commit one complete generation, discarding every previous state version."""
        if (latents is None) != (feats is None):
            raise ValueError("History latents and features must be supplied together")
        if latents is not None and (latents.shape[1] != len(cams) or feats.shape[1] != len(cams)):
            raise ValueError("History cameras, video and features must align by view")
        self.gaussians, self.cams = [gaussians], [cams.detach()]
        self.latents = [None if latents is None else latents.detach().clone()]
        self.feats = [None if feats is None else feats.detach().float().clone()]

    def all_gaussians(self, detach=False, upto=None):
        gs = self.gaussians[:upto] if upto is not None else self.gaussians
        if not gs:
            return None
        result = gs[0].detach() if detach else gs[0]
        return result

    def all_cams(self):
        return self.cams[0]

    def history(self):
        if not self.cams:
            return None, None, None
        if self.latents[0] is None or self.feats[0] is None:
            raise ValueError("The state does not contain a complete rollout history")
        return self.cams[0], self.latents[0], self.feats[0]


class ARFlashWorld:
    """Wraps the FlashWorld generator (1174-ch DiT with feedback) and the 3DGS decoder for chunk-wise
    autoregressive generation. Batch size is 1 (one scene / one session)."""

    def __init__(self, generator, recon_decoder, codec, timesteps, denoising_steps=DEFAULT_DENOISING_STEPS,
                 vae_micro_batch=8, vae_checkpoint=True,
                 recon_micro_batch=None, recon_checkpoint=True, vae_micro_batch_nograd=None,
                 recon_micro_batch_nograd=None):
        self.generator = generator
        self.recon = recon_decoder
        self.codec = codec
        self.timesteps = timesteps
        self.denoising_steps = list(denoising_steps)
        self.vae_micro_batch = vae_micro_batch
        self.vae_micro_batch_nograd = vae_micro_batch_nograd or vae_micro_batch
        self.vae_checkpoint = vae_checkpoint
        self.recon_micro_batch = recon_micro_batch
        self.recon_micro_batch_nograd = recon_micro_batch_nograd or recon_micro_batch
        self.recon_checkpoint = recon_checkpoint

    @property
    def num_steps(self):
        return len(self.denoising_steps)

    def step_t(self, i):
        return self.timesteps[self.denoising_steps[i]]

    # -- rendering ---------------------------------------------------------------------------
    def render(self, gaussians, cams, height, width, bg_mode="white"):
        """gaussians [N,38], cams [V,11] -> rgb [1,V,3,H,W] in [-1,1], depth [1,V,1,H,W]."""
        return self.recon.render([gaussians], cams[None], height, width, bg_mode=bg_mode)

    def render_latents(self, gaussians, cams, height, width, bg_mode="white"):
        rgb, depth = self.render(gaussians, cams, height, width, bg_mode)
        mb = self.vae_micro_batch if torch.is_grad_enabled() else self.vae_micro_batch_nograd
        lat = self.codec.encode(rgb, use_checkpoint=self.vae_checkpoint, micro_batch=mb)
        return lat, rgb, depth

    @torch.no_grad()
    def _render_latents_nograd(self, gs, cams, height, width):
        rgb, _ = self.render(gs, cams, height, width, "white")
        return self.codec.encode(rgb, micro_batch=self.vae_micro_batch_nograd)

    def decode_gaussians(self, feats, x0, local_cams):
        """feats [1,1024,K,h,w], x0 [1,48,K,h,w] (scaled), local_cams [K,11] -> [K*P*h'*w', 38].
        The decoder works per view, so views can be decoded in micro-batches (bounds the recompute peak)."""
        f = einops.rearrange(feats, "B C T H W -> (B T) C H W").unsqueeze(2)
        z = einops.rearrange(self.codec.unscale(x0), "B C T H W -> (B T) C H W").unsqueeze(2)
        mb = (self.recon_micro_batch if torch.is_grad_enabled() else self.recon_micro_batch_nograd) or f.shape[0]
        out = []
        for fi, zi, ci in zip(f.split(mb), z.split(mb), local_cams.split(mb)):
            if self.recon_checkpoint and torch.is_grad_enabled():
                gi = checkpoint(self.recon, fi, zi, ci[None], use_reentrant=False)
            else:
                gi = self.recon(fi, zi, ci[None])
            out.append(gi.flatten(1, -2)[0])
        return torch.cat(out, 0)

    # -- one DiT call --------------------------------------------------------------------------
    def _dit(self, x, raymaps, cond, prev_feat, prev_lat, t, text_emb, hist_lat=None, hist_feat=None):
        """Apply bidirectional attention to the complete prefix at per-view noise levels."""
        n_h = 0 if hist_lat is None else hist_lat.shape[2]
        if n_h > 0:
            tokens_per_frame = (x.shape[-2] // PATCH_HW) * (x.shape[-1] // PATCH_HW)
            x = torch.cat([hist_lat, x], 2)
            # History slots condition on completed direct latents. Current
            # slots contain observations only at the supplied reference poses.
            cond = torch.cat([hist_lat, cond], 2)
            prev_feat = torch.cat([hist_feat, prev_feat], 2)
            prev_lat = torch.cat([hist_lat, prev_lat], 2)
            timestep = torch.cat([t.new_zeros(n_h * tokens_per_frame),
                                  t.reshape(1).expand((x.shape[2] - n_h) * tokens_per_frame)])[None]
        else:
            timestep = t.reshape(1)
        hidden = torch.cat([x, raymaps, cond, prev_feat, prev_lat], 1)
        out = self.generator(hidden_states=hidden, timestep=timestep, encoder_hidden_states=text_emb, return_dict=False)[0]
        v, feats = out.split([LATENT_DIM, FEAT_DIM], 1)
        return v[:, :, n_h:].float(), feats

    # -- one chunk -----------------------------------------------------------------------------
    @torch.no_grad()
    def generate_chunk(self, world: ARGaussianWorld, cams, text_emb, height, width, cond_latents=None):
        """Denoise a chunk and update the complete scene state."""
        device = cams.device
        K = cams.shape[0]
        h, w = height // SPATIAL_DOWN, width // SPATIAL_DOWN
        hist_cams, hist_lat, hist_feat = world.history()
        n_h = 0 if hist_cams is None else len(hist_cams)
        ctx_cams = cams if n_h == 0 else torch.cat([hist_cams, cams], 0)
        local_cams, origin, scale = local_normalize(ctx_cams)
        raymaps = raymaps_for(local_cams, h, w)
        hist_lat = None if n_h == 0 else hist_lat[None]
        hist_feat = None if n_h == 0 else hist_feat[None]

        x = torch.randn(1, LATENT_DIM, K, h, w, device=device)
        cond = torch.zeros_like(x) if cond_latents is None else cond_latents
        prev_lat = torch.zeros_like(x)
        prev_feat = torch.zeros(1, FEAT_DIM, K, h, w, device=device)
        with torch.no_grad():
            for i in range(self.num_steps):
                t = self.step_t(i)
                v, feats = self._dit(x, raymaps, cond, prev_feat, prev_lat, t, text_emb, hist_lat, hist_feat)
                x0_2d = x - (t / 1000.0) * v
                decode_lat = torch.cat([hist_lat, x0_2d], 2) if n_h else x0_2d
                gs = gaussians_to_global(self.decode_gaussians(feats, decode_lat, local_cams), origin, scale)
                if i < self.num_steps - 1:
                    prev_lat = self._render_latents_nograd(gs, cams, height, width)
                    prev_feat = feats[:, :, n_h:].float()
                    sigma_next = self.step_t(i + 1) / 1000.0
                    x = (1 - sigma_next) * prev_lat + sigma_next * torch.randn_like(prev_lat)

        # The complete direct RGB history stays clean. Geometry and features
        # are replaced by the most recent reconstruction of the whole prefix.
        world.replace_all(gs, ctx_cams, decode_lat[0], feats[0])
        context_x0 = torch.cat([hist_lat, x0_2d], 2) if n_h else x0_2d
        return {"gaussians": gs, "x0_2d": x0_2d, "x0_2d_context": context_x0,
                "num_history": n_h, "num_refreshed_views": n_h,
                "history_indices": list(range(n_h)), "num_decoded_views": len(ctx_cams)}

    def rollout(self, cams, chunk_size, text_emb, height, width, image_latent=None, world=None,
                on_chunk=None, return_outputs=True):
        """Generate a trajectory chunk by chunk, optionally continuing an existing world."""
        if chunk_size < 1 or len(cams) < 1:
            raise ValueError("Expected a positive chunk size and at least one camera")
        world = ARGaussianWorld() if world is None else world
        h, w = height // SPATIAL_DOWN, width // SPATIAL_DOWN
        initial_cond = None
        if image_latent is not None and world.num_views == 0:
            refs = image_latent.unsqueeze(2) if image_latent.ndim == 4 else image_latent
            first_size = min(chunk_size, len(cams))
            if (refs.ndim != 5 or refs.shape[:2] != (1, LATENT_DIM)
                    or refs.shape[-2:] != (h, w) or not 1 <= refs.shape[2] <= first_size):
                raise ValueError("image_latent must be [1,48,h,w] or [1,48,R,h,w], with R fitting the first chunk")
            initial_cond = torch.zeros(1, LATENT_DIM, first_size, h, w, device=cams.device)
            initial_cond[:, :, :refs.shape[2]] = refs.to(device=cams.device)
        outs = []
        for cams_b in cams.split(chunk_size, 0):
            cond = None
            if world.num_views == 0:
                cond = initial_cond
            output = self.generate_chunk(world, cams_b, text_emb, height, width, cond)
            if return_outputs:
                outs.append(output)
            del output
            if on_chunk is not None:
                on_chunk(world)
        return world, outs


class ARWorldSession:
    """Interactive inference: start a world from a first trajectory, then keep extending it."""

    def __init__(self, model: ARFlashWorld, text_emb, height, width, chunk_size, image_latent=None):
        self.model = model
        self.text_emb = text_emb
        self.height, self.width = height, width
        self.chunk_size = chunk_size
        self.image_latent = image_latent
        self.world = ARGaussianWorld()
        self.ref_w2c = None
        self.T_norm = None

    @torch.no_grad()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def to_global(self, cams_raw):
        """cams_raw [N,11] in the dataset / user frame -> session frame (fixed by the first trajectory)."""
        if self.ref_w2c is None:
            # Establish the session frame from every camera in the first input
            # segment. Later segments reuse the saved frame and extent.
            cams, self.ref_w2c, self.T_norm = normalize_cameras(cams_raw[None].float(), return_meta=True)
        else:
            cams = normalize_cameras(cams_raw[None].float(), ref_w2c=self.ref_w2c, T_norm=self.T_norm)
        return cams[0]

    @torch.no_grad()
    def extend(self, cams_raw, return_outputs=False):
        if cams_raw.ndim != 2 or cams_raw.shape[1] != 11 or not len(cams_raw):
            raise ValueError("Expected a nonempty camera tensor of shape [N, 11]")
        cams = self.to_global(cams_raw)
        image_latent = self.image_latent if self.world.num_views == 0 else None
        _, outputs = self.model.rollout(
            cams, self.chunk_size, self.text_emb, self.height, self.width, image_latent,
            world=self.world, return_outputs=return_outputs)
        self.image_latent = None
        return (cams, outputs) if return_outputs else cams

    def state_dict(self):
        """Return the tensors required to resume at the next chunk boundary."""
        if self.world.num_views == 0:
            raise ValueError("Generate at least one chunk before saving a session")
        cams, latents, features = self.world.history()
        return {
            "version": 1,
            "height": self.height,
            "width": self.width,
            "chunk_size": self.chunk_size,
            "text_emb": None if self.text_emb is None else self.text_emb.detach().cpu(),
            "cameras": cams.cpu(),
            "rgb_latents": latents.cpu(),
            "features": features.cpu(),
            "gaussians": self.world.all_gaussians(detach=True).cpu(),
            "reference_w2c": self.ref_w2c.cpu(),
            "translation_extent": self.T_norm.cpu(),
        }

    @classmethod
    def from_state_dict(cls, model, state, device):
        """Restore a completed state with the same model and camera convention."""
        if state.get("version") != 1:
            raise ValueError("Unsupported ArgsWorld session version")
        height, width, chunk_size = (state[k] for k in ("height", "width", "chunk_size"))
        if height < 32 or width < 32 or height % 32 or width % 32 or chunk_size < 1:
            raise ValueError("Invalid saved image dimensions or chunk size")
        text_emb = state["text_emb"]
        session = cls(model, None if text_emb is None else text_emb.to(device), height, width, chunk_size)
        cameras, latents, features, gaussians, ref_w2c, extent = (
            state[key] for key in ("cameras", "rgb_latents", "features", "gaussians",
                                   "reference_w2c", "translation_extent")
        )
        n = len(cameras)
        h, w = height // SPATIAL_DOWN, width // SPATIAL_DOWN
        if (
            not n or cameras.shape != (n, 11)
            or latents.shape != (LATENT_DIM, n, h, w) or features.shape != (FEAT_DIM, n, h, w)
            or gaussians.ndim != 2 or gaussians.shape[1] != 38 or len(gaussians) == 0
            or ref_w2c.shape != (1, 1, 4, 4) or extent.shape != (1, 1, 1)
        ):
            raise ValueError("Inconsistent saved camera, history, or Gaussian shapes")
        session.ref_w2c, session.T_norm = ref_w2c.to(device), extent.to(device)
        session.world.replace_all(gaussians.to(device), cameras.to(device),
                                  latents.to(device), features.to(device))
        return session


def interpolate_cameras(cams, frames_per_view=4):
    from utils import sample_from_dense_cameras
    n = cams.shape[0]
    if n < 2 or frames_per_view == 1:
        return cams
    t = torch.linspace(0, 1, (n - 1) * frames_per_view + 1)
    return sample_from_dense_cameras(cams.float().cpu(), t).to(cams.device)


@torch.no_grad()
def render_frames(ar: ARFlashWorld, gaussians, cams, height, width, batch=8):
    """-> uint8 numpy [T,H,W,3] rendered with a white background."""
    frames = []
    for c in cams.split(batch, 0):
        rgb, _ = ar.render(gaussians, c, height, width, "white")
        frames.append(((rgb[0].clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu())
    return torch.cat(frames).numpy()


def write_video(path, frames, fps=24):
    import imageio
    imageio.mimwrite(path, list(frames), fps=fps, quality=8, macro_block_size=1)


def save_rows(path, rows, max_cols=8, downsample=2):
    """rows: list of [V,3,H,W] in [-1,1] (same V) -> one image grid, `max_cols` evenly spaced views."""
    import torchvision
    V = rows[0].shape[0]
    idx = torch.linspace(0, V - 1, min(V, max_cols)).round().long()
    picked = [r[idx.to(r.device)].float().cpu() for r in rows]
    grid = torch.cat(picked, 0)
    if downsample > 1:
        grid = F.avg_pool2d(grid, downsample)
    torchvision.utils.save_image((grid + 1) / 2, path, nrow=len(idx), padding=2, pad_value=1.0)
