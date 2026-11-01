import unittest

import torch

from hier_video.models.semantic_decoder import GOPDecoder
from hier_video.models.semantic_encoder import GOPEncoder
from hier_video.transmission.reference_delivery import deliver
from hier_video.transmission.reference_layers import restore_layers, split_layers
from hier_video.transmission.reference_transport import make_offer, receive


class CnnBridgeTests(unittest.TestCase):
    def test_same_configuration_matches_after_byte_transport(self):
        torch.manual_seed(24)
        torch.set_num_threads(1)
        encoder = GOPEncoder(24).eval()
        decoder = GOPDecoder(24).eval()
        video = torch.rand(1, 4, 3, 32, 32)
        with torch.no_grad():
            feature = encoder(video)
            parts = split_layers(feature)
            offer = make_offer(parts, session_id=31, gop_id=0, model_id="test-cnn")
            for mode in ("base", "full"):
                direct = decoder(restore_layers(parts, mode))
                reception = receive(deliver(offer, mode).received)
                received_rgb = decoder(reception.features)
                self.assertEqual(received_rgb.shape, video.shape)
                self.assertTrue(torch.equal(direct, received_rgb))


if __name__ == "__main__":
    unittest.main()
