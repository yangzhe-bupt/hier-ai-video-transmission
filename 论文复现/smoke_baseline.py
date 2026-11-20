"""Small CPU forward/backward and transport-interface check. No training."""
from pathlib import Path
import sys
import torch

root = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
for relative in ('practice/models', 'practice/models/swin', 'Biran', 'packet'):
    sys.path.insert(0, str(root / relative))
from swin_encoder import SwinGOPEncoder
from semantic_decoder import GOPDecoder
from feature_decomposition import PublicResidualDecomposer
from packet_loss import simulate_packet_loss

torch.manual_seed(42)
torch.set_num_threads(2)
encoder = SwinGOPEncoder()
decoder = GOPDecoder(feature_channels=24)
video = torch.rand(1, 4, 3, 32, 32)
features = encoder(video)
parts = PublicResidualDecomposer(downsample_factor=2).encode(features)
restored = PublicResidualDecomposer(downsample_factor=2).decode(parts)
received, missing, dropped = simulate_packet_loss(restored, 0.25, 16)
output = decoder(received)
assert output.shape == video.shape
assert torch.isfinite(output).all()
assert torch.equal(received[~missing], restored[~missing])
assert (received[missing] == 0).all()
for rate in (0.0, 1.0):
    value, mask, _ = simulate_packet_loss(restored, rate, 16)
    assert mask.all() if rate == 1.0 else not mask.any()
    assert (value == 0).all() if rate == 1.0 else torch.equal(value, restored)
loss = (output - video).square().mean()
loss.backward()
for model in (encoder, decoder):
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
print('PASS: encoder -> decomposition -> packet loss -> decoder -> backward')
print('feature shape:', tuple(features.shape))
print('MSE (random weights, not a quality result):', loss.item())
print('torch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())
