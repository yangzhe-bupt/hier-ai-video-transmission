import tempfile
from pathlib import Path
import torch
from hier_video.models.semantic_encoder import GOPEncoder
from hier_video.models.semantic_decoder import GOPDecoder
from hier_video.checkpoint import save_checkpoint, load_models

def check_autoencoder(device):
    """检查联合梯度、参数更新及保存/加载，不使用私有视频。"""
    torch.manual_seed(24)
    encoder=GOPEncoder(24).to(device)
    decoder=GOPDecoder(24).to(device)
    x=torch.rand(1,4,3,32,32,device=device)
    optimizer=torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()),lr=1e-3)
    before=[p.detach().clone() for p in encoder.parameters()]
    optimizer.zero_grad()
    z=encoder(x)
    y=decoder(z)
    assert z.shape==(1,4,24,8,8) and y.shape==x.shape
    assert y.min()>=0 and y.max()<=1
    loss=(y-x).square().mean()
    loss.backward()
    for p in list(encoder.parameters())+list(decoder.parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all()
    optimizer.step()
    assert any(not torch.equal(a,b) for a,b in zip(before,encoder.parameters()))
    encoder.eval();decoder.eval()
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'checkpoint.pt'
        save_checkpoint(encoder,decoder,[loss.item()],path)
        enc,dec=load_models(path,device)
        with torch.no_grad():
            assert torch.allclose(dec(enc(x)),decoder(encoder(x)),rtol=1e-5,atol=1e-6)
    try:
        encoder(torch.rand(1,4,3,30,32,device=device))
    except ValueError:
        pass
    else:
        raise AssertionError('不符合总下采样倍数的尺寸应被拒绝')
    print(f'{device}: 编解码梯度、更新、尺寸、检查点检查通过')

def main():
    check_autoencoder('cpu')
    if torch.cuda.is_available():
        check_autoencoder('cuda')

if __name__=='__main__':
    main()
