"""Explicit receiver state and feature-domain missing masks for int8 packets.

The packet reassembler deliberately fills absent regions with zero.  Call this
audit before dequantization/decoding so zero fill is never mistaken for a
complete reception.
"""

from dataclasses import dataclass

import torch

from .feature_mapping import FeatureGroupManifest
from .packetization import UnpackedFeatureParts


@dataclass(frozen=True)
class ReceptionAudit:
    status: str
    decoder_ready: bool
    missing_public_group_ids: tuple[int, ...]
    missing_individual_group_ids: tuple[int, ...]
    public_missing_mask: torch.Tensor
    individual_missing_mask: torch.Tensor
    feature_missing_mask: torch.Tensor


def audit_quantized_reception(
    received: UnpackedFeatureParts, manifest: FeatureGroupManifest
) -> ReceptionAudit:
    """Classify reception; all masks use BTCHW and True means missing.

    A partial individual branch may be decoded with zero residual at missing
    positions, provided its scale is known.  Any missing public region fails
    because it affects every frame; a missing branch scale also fails.
    """
    manifest.validate()
    if tuple(received.public.shape) != manifest.public_shape or tuple(
        received.individual.shape
    ) != manifest.individual_shape:
        raise ValueError("Received branch shapes do not match the manifest")
    known = {group.group_id for group in manifest.groups}
    missing = set(received.missing_group_ids)
    if not missing <= known:
        raise ValueError("Unknown missing group ID")

    public_mask = torch.zeros(manifest.public_shape, dtype=torch.bool, device=received.public.device)
    individual_mask = torch.zeros(
        manifest.individual_shape, dtype=torch.bool, device=received.individual.device
    )
    missing_public = []
    missing_individual = []
    for group in manifest.groups:
        if group.group_id not in missing:
            continue
        target = public_mask if group.branch == "public" else individual_mask
        region = group.tensor_region
        target[
            region.b[0] : region.b[1],
            region.t[0] : region.t[1],
            region.c[0] : region.c[1],
            region.h[0] : region.h[1],
            region.w[0] : region.w[1],
        ] = True
        (missing_public if group.branch == "public" else missing_individual).append(
            group.group_id
        )

    metadata = received.quantization_by_branch
    if missing_public:
        status = "failed_public_incomplete"
    elif "public" not in metadata or "individual" not in metadata:
        status = "failed_quantization_config_missing"
    elif missing_individual:
        status = "degraded_individual_zero_fill"
    else:
        status = "complete"
    feature_mask = public_mask.expand_as(individual_mask) | individual_mask
    return ReceptionAudit(
        status=status,
        decoder_ready=status in ("complete", "degraded_individual_zero_fill"),
        missing_public_group_ids=tuple(missing_public),
        missing_individual_group_ids=tuple(missing_individual),
        public_missing_mask=public_mask,
        individual_missing_mask=individual_mask,
        feature_missing_mask=feature_mask,
    )
