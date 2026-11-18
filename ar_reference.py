"""Reference images occupy known camera slots in the first AR chunk."""


def validate_reference_indices(indices, first_chunk_size):
    indices = list(indices)
    if not 1 <= len(indices) <= min(8, first_chunk_size):
        raise ValueError(f"Expected 1 to {min(8, first_chunk_size)} reference images, got {len(indices)}")
    if any(type(i) is not int or not 0 <= i < first_chunk_size for i in indices):
        raise ValueError(f"Reference indices must be integers in [0, {first_chunk_size})")
    if len(set(indices)) != len(indices):
        raise ValueError("Reference indices must be unique")
    return indices


def pack_reference_latents(latents, indices):
    """Place the encoded references in their calibrated first-chunk slots."""
    if latents.ndim != 5 or latents.shape[0] != 1 or latents.shape[2] != len(indices):
        raise ValueError("Reference latents must have shape [1,C,len(indices),h,w]")
    validate_reference_indices(indices, max(indices, default=-1) + 1)
    if indices == [0]:
        return latents[:, :, 0]
    packed = latents.new_zeros(1, latents.shape[1], max(indices) + 1, *latents.shape[-2:])
    packed[:, :, indices] = latents
    return packed


def add_reference_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--num_reference_frames", type=int, default=None,
                       help="Use the first N camera slots for the references (1 to 8)")
    group.add_argument("--reference_indices", type=int, nargs="+", default=None,
                       help="Reference camera slots in the first chunk (zero-based, e.g. 0 3 7)")


def inference_reference_indices(args, first_chunk_size, image_count=None):
    indices = getattr(args, "reference_indices", None)
    count = getattr(args, "num_reference_frames", None)
    if indices is not None and count is not None:
        raise ValueError("Use either --reference_indices or --num_reference_frames")
    if indices is None:
        count = count if count is not None else (image_count if image_count is not None else 1)
        if not 1 <= count <= min(8, first_chunk_size):
            raise ValueError("Reference count must fit the first chunk and cannot exceed eight")
        indices = list(range(count))
    indices = validate_reference_indices(indices, first_chunk_size)
    if image_count is not None and len(indices) != image_count:
        raise ValueError("The number of images must match the number of reference camera slots")
    return indices
