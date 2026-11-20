"""PING-III transmission components."""

from .network_replay import FeaturePacket, LinkSample, replay_feature_packet_events
from .reference_layers import ReferenceBackend, restore_layers, split_layers
from .reference_transport import make_offer, receive, selected_packets
from .feature_decomposition import (
    FeatureMeta,
    FeatureParts,
    PublicResidualDecomposer,
    parts_payload_bytes,
)
from .feature_mapping import (
    FeatureGroup,
    FeatureGroupManifest,
    TensorRegion,
    build_group_manifest,
)
from .packetization import (
    FeaturePacket as SerializedFeaturePacket,
    PacketHeader,
    UnpackedFeatureParts,
    build_feature_packets,
    deserialize_packet,
    unpack_feature_packets,
)
from .quantization import (
    QuantizedTensor,
    dequantize_from_metadata,
    dequantize_symmetric,
    quantize_symmetric,
    quantized_payload_bytes,
)

__all__ = [
    "FeaturePacket",
    "LinkSample",
    "ReferenceBackend",
    "make_offer",
    "receive",
    "replay_feature_packet_events",
    "restore_layers",
    "selected_packets",
    "split_layers",
    "FeatureMeta",
    "FeatureParts",
    "PublicResidualDecomposer",
    "parts_payload_bytes",
    "FeatureGroup",
    "FeatureGroupManifest",
    "TensorRegion",
    "build_group_manifest",
    "SerializedFeaturePacket",
    "PacketHeader",
    "UnpackedFeatureParts",
    "build_feature_packets",
    "deserialize_packet",
    "unpack_feature_packets",
    "QuantizedTensor",
    "quantize_symmetric",
    "dequantize_symmetric",
    "dequantize_from_metadata",
    "quantized_payload_bytes",
]
