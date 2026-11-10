import sys, torch, argparse, math
sys.path.insert(0, '.')
from models.zoo import build_model, ZOO

torch.manual_seed(0)
a = argparse.Namespace(seq_len=96, label_len=48, pred_len=96, enc_in=7, d_model=32,
                       d_ff=64, n_heads=4, e_layers=2, d_layers=1, dropout=0.1,
                       factor=3, patch_len=16, stride=8, modes=32, top_k=3,
                       moving_avg=25, cycle_length=24,
                       raft_topk=3, tfps_patterns=4, tfps_tau=1.0,
                       stmtm_kernel=25, stmtm_mask_ratio=0.4)

# --- synthetic multivariate series with 3 sinusoids per channel -----------------
t = torch.arange(0, 96 + 96 + 700).float()
cols = []
for ch in range(7):
    s = torch.sin(2 * math.pi * t / (20 + 3 * ch)) + 0.3 * torch.sin(2 * math.pi * t / (7 + ch))
    cols.append(s)
X = torch.stack(cols, -1) + 0.05 * torch.randn(len(t), 7)
win = [X[i:i + 192] for i in range(0, 500)]
xb = torch.stack([w[:96] for w in win])
yb = torch.stack([w[96:192] for w in win])
print('data', xb.shape, yb.shape)

for name in ['tfps', 'stmtm', 'raft']:
    m = build_model(name, a)
    n = sum(p.numel() for p in m.parameters())
    # RAFT bank: reuse the training windows, exactly as the protocol does
    if hasattr(m, 'set_retrieval_bank'):
        m.set_retrieval_bank(xb.transpose(1, 2), yb)
    out = m(xb[:4])
    assert out.shape == (4, 96, 7), (name, out.shape)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    losses = []
    extra = ''
    if hasattr(m, 'ssl_loss'):
        for _ in range(30):
            idx = torch.randint(0, len(xb), (16,))
            loss = m.ssl_loss(xb[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        extra = f' ssl={float(loss):.4f}'
    for step in range(200):
        idx = torch.randint(0, len(xb), (16,))
        out = m(xb[idx])
        loss = torch.nn.functional.mse_loss(out, yb[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    ok = not any(map(math.isnan, losses))
    print(f'{name:8s} params={n:>8,} out={tuple(out.shape)} '
          f'loss {losses[0]:.4f} -> {sum(losses[-10:]) / 10:.4f} nan={not ok}{extra}')
print('OK')
