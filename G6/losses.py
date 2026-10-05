# -*- coding: utf-8 -*-
"""Ham mat mat, muc 3.3 de cuong.

  CT (7): L = L_gap + lambda * L_boundary,  L_gap = (1/128) * sum_{t=a}^{b} (z_hat_t - z_t)^2
  CT (8): L_boundary = (1/|B|) * sum_{t in B} [ (z_out_{t+1} - z_out_t) - (z_{t+1} - z_t) ]^2
  CT (9): B = {a-1, a, b-1, b},  b = a + 127

Trong do z_out = (1 - M) z + M z_hat theo cong thuc (5). So hang (8) doi chieu do doc tai tao
voi do doc dich, khong ep do doc ve 0. Tap B gom hai cap noi voi vung quan sat (a-1,a) va (b,b+1)
cung hai cap phia trong khoang thieu (a,a+1) va (b-1,b). Tinh tren mien chuan hoa, lay trung binh
trong tung cua so roi trung binh theo batch. Moi cua so co the co a khac nhau.
"""
import torch

GAP = 128
B_OFFSET = torch.tensor([-1, 0, GAP - 2, GAP - 1])     # a-1, a, b-1, b  voi b = a + 127


def l_gap(z_hat, z, M):
    """CT (7): MSE trung binh tren dung 128 mau bi che, tung cua so roi trung binh batch."""
    e2 = ((z_hat - z) ** 2) * M
    return (e2.sum(dim=1) / GAP).mean()


def l_boundary(z_hat, z, M, a):
    """CT (8) va (9). a: (B,) chi so bat dau khoang thieu cua tung cua so."""
    z_out = (1.0 - M) * z + M * z_hat                    # CT (5)
    d_out = z_out[:, 1:] - z_out[:, :-1]                 # d[t] = z_out[t+1] - z_out[t]
    d_z = z[:, 1:] - z[:, :-1]
    idx = a.unsqueeze(1) + B_OFFSET.to(a.device).unsqueeze(0)      # (B, 4)
    # Khi khoang thieu cham mau cuoi cua so (b = 511, chi co o bien the cai tien do tre) thi cap (b, b+1) khong ton tai:
    # bo cap do, lay trung binh tren cac cap con lai. Voi a <= 320 ca 4 cap deu hop le nen ket qua y het truoc.
    hop_le = idx < d_out.shape[1]
    idx = idx.clamp(max=d_out.shape[1] - 1)
    sai = (torch.gather(d_out, 1, idx) - torch.gather(d_z, 1, idx)) * hop_le   # (B, 4)
    return ((sai ** 2).sum(dim=1) / hop_le.sum(dim=1)).mean()


def loss_inpaint(z_hat, z, M, a, lam):
    lg = l_gap(z_hat, z, M)
    lb = l_boundary(z_hat, z, M, a)
    return lg + lam * lb, lg.detach(), lb.detach()


if __name__ == '__main__':
    import data as D
    torch.manual_seed(0)
    Bn, T = 4, 512
    a = torch.tensor([64, 192, 320, 100])
    M = torch.zeros(Bn, T)
    for i in range(Bn):
        M[i, a[i]: a[i] + GAP] = 1
    z = torch.randn(Bn, T)

    # 1) du doan dung tuyet doi -> ca hai so hang bang 0
    l, lg, lb = loss_inpaint(z.clone(), z, M, a, lam=1.0)
    print('du doan = dich          : L=%.6f  L_gap=%.6f  L_boundary=%.6f' % (l, lg, lb))

    # 2) sai o ngoai khoang thieu khong anh huong (do M)
    zh = z.clone()
    zh[:, :50] += 100.0
    l, lg, lb = loss_inpaint(zh, z, M, a, lam=1.0)
    print('sai ngoai khoang thieu  : L=%.6f  (phai bang 0)' % l)

    # 3) do doc tai bien: dich thang, khong ep ve 0. Dich dung, chi lech mot hang so trong gap
    zh = z.clone()
    for i in range(Bn):
        zh[i, a[i]: a[i] + GAP] += 1.0
    l, lg, lb = loss_inpaint(zh, z, M, a, lam=1.0)
    # trong gap do doc khong doi, chi hai cap noi voi vung quan sat lech 1 -> (1^2 + 1^2)/4 = 0.5
    print('lech hang so 1 trong gap: L_gap=%.4f (phai 1.0)  L_boundary=%.4f (phai 0.5)' % (lg, lb))

    # 4) doi chieu voi tinh tay bang numpy cho mot cua so
    import numpy as np
    i = 1
    zn, zhn, Mn = z[i].numpy(), zh[i].numpy(), M[i].numpy()
    zo = (1 - Mn) * zn + Mn * zhn
    aa = int(a[i]); bb = aa + GAP - 1
    tay = np.mean([((zo[t + 1] - zo[t]) - (zn[t + 1] - zn[t])) ** 2 for t in (aa - 1, aa, bb - 1, bb)])
    lb1 = l_boundary(zh[i:i + 1], z[i:i + 1], M[i:i + 1], a[i:i + 1]).item()
    print('L_boundary mot cua so: torch=%.6f  numpy=%.6f  %s' % (lb1, tay, 'OK' if abs(lb1 - tay) < 1e-6 else 'LECH'))
