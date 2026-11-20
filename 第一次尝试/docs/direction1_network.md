# Direction-one network integration

This package adds the network-side reference needed to connect the existing
CNN and `Biran/feature_decomposition.py` components. It is a functional
integration baseline. It is not a trained codec, a real socket implementation,
or a claim that the project target has been reached.

## What is included

- `hier_video.transmission.network_replay`
  - deterministic packet events for `received`, `lost`, `late`, and `unsent`;
  - integer serialization-time accounting;
  - selected and actual wire-byte totals;
  - deadline and budget validation;
  - stable per-packet loss draws for comparable policies.
- `hier_video.transmission.reference_layers`
  - a replaceable reference backend for two actions: `base` and `full`;
  - temporal public feature plus pooled residual;
  - explicit rejection of enhancement-only decoding.
- `hier_video.transmission.reference_transport`
  - experimental float32 fragment format with in-band configuration;
  - packet identity, offsets, CRC, branch hashes, and receiver-only masks;
  - missing-base handling that disables the enhancement branch.
- `hier_video.transmission.reference_delivery`
  - binding between the reference datagrams and the deterministic event
    replay.

The public/individual mapping is kept as a reference mapping. It must be
confirmed by the codec and recovery owners before it is called a formal
base/enhancement-layer protocol.

## Minimal example

```python
import torch

from hier_video.models.semantic_encoder import GOPEncoder
from hier_video.transmission import make_offer, receive, split_layers
from hier_video.transmission.reference_delivery import deliver

encoder = GOPEncoder(embed_dim=24).eval()
gop = torch.rand(1, 4, 3, 32, 32)
features = encoder(gop)
parts = split_layers(features, factor=2)
offer = make_offer(parts, session_id=1, gop_id=0, model_id="example")

transfer = deliver(offer, mode="base", loss=0.1, seed=7)
received = receive(transfer.received)
print(received.state, transfer.report["actual_wire_bytes"])
```

## Run checks

From the repository root:

```powershell
python checks/check_direction1_network.py
python -m unittest discover -s tests/direction1 -v
```

The reference transport counts its in-band configuration and configured
IPv4/UDP example overhead. It does not claim Ethernet, Wi-Fi, cellular
physical-layer, ACK, FEC, retransmission, or model-weight costs. It also does
not include quantization or entropy coding.
