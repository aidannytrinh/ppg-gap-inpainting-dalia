# -*- coding: utf-8 -*-
"""Mo hinh tu ma hoa co mat na 1D-CNN + Bi-LSTM, dung bang muc 3.1 de cuong.

  Khoi        Cau hinh                              Dau ra (T, C)   Tham so
  Ma hoa 1    Conv1D 2->16, k=7; ReLU; Pool 2        (256, 16)         240
  Ma hoa 2    Conv1D 16->32, k=7; ReLU; Pool 2       (128, 32)       3.616
  Ma hoa 3    Conv1D 32->64, k=7; ReLU; Pool 2       (64, 64)       14.400
  Bi-LSTM     64 dau vao; 64 an moi chieu; 1 tang    (64, 128)      66.560
  Giai ma 1   Up 2; Conv1D 128->32, k=7; ReLU        (128, 32)      28.704
  Giai ma 2   Up 2; Conv1D 32->16, k=7; ReLU         (256, 16)       3.600
  Giai ma 3   Up 2; Conv1D 16->1, k=7; tuyen tinh    (512, 1)          113
  Tong                                                (512, 1)     117.233

Conv1D stride 1, padding 3, bias=True. MaxPool kernel va stride 2. Upsample kieu nearest.
Bi-LSTM bias=True, khong projection, khong them tang chuan hoa. Chuyen truc truoc va sau LSTM.
Lop dau ra tuyen tinh (khong ReLU) de du doan duoc gia tri chuan hoa am lan duong.

Ban bo Bi-LSTM noi (64, 64) thang vao giai ma: 36.337 tham so.
Ban them ba truc ACC (dau vao 5 kenh): 117.569 tham so.

Thu tu truc: mo hinh nhan (B, T, C) theo cach ghi cua de cuong va tu chuyen sang (B, C, T)
cho Conv1d cua PyTorch.
"""
import torch
import torch.nn as nn

K, PAD = 7, 3


class InpaintAE(nn.Module):
    def __init__(self, in_ch=2, dung_lstm=True):
        super().__init__()
        self.dung_lstm = dung_lstm
        self.ma_hoa = nn.Sequential(
            nn.Conv1d(in_ch, 16, K, stride=1, padding=PAD, bias=True), nn.ReLU(), nn.MaxPool1d(2, 2),
            nn.Conv1d(16, 32, K, stride=1, padding=PAD, bias=True), nn.ReLU(), nn.MaxPool1d(2, 2),
            nn.Conv1d(32, 64, K, stride=1, padding=PAD, bias=True), nn.ReLU(), nn.MaxPool1d(2, 2))
        self.lstm = nn.LSTM(64, 64, num_layers=1, batch_first=True, bidirectional=True,
                            bias=True, proj_size=0) if dung_lstm else None
        c = 128 if dung_lstm else 64
        self.giai_ma = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv1d(c, 32, K, 1, PAD, bias=True), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv1d(32, 16, K, 1, PAD, bias=True), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv1d(16, 1, K, 1, PAD, bias=True))

    def forward(self, u):
        """u: (B, 512, C) -> z_hat: (B, 512)."""
        h = self.ma_hoa(u.permute(0, 2, 1))          # (B, 64, 64) theo (B, C, T)
        if self.dung_lstm:
            h, _ = self.lstm(h.permute(0, 2, 1))     # LSTM can (B, T, C)
            h = h.permute(0, 2, 1)                   # ve lai (B, C, T)
        return self.giai_ma(h).squeeze(1)            # (B, 512)


def dem_tham_so(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == '__main__':
    cau_hinh = [('CNN-Bi-LSTM, 2 kenh', 2, True, 117_233),
                ('CNN khong Bi-LSTM, 2 kenh', 2, False, 36_337),
                ('CNN-Bi-LSTM, them 3 truc ACC', 5, True, 117_569)]
    for ten, c, l, mong_doi in cau_hinh:
        m = InpaintAE(c, l)
        n = dem_tham_so(m)
        with torch.no_grad():
            y = m(torch.zeros(2, 512, c))
        print('%-32s tham so = %7d (de cuong %7d) %s | dau ra %s'
              % (ten, n, mong_doi, 'OK' if n == mong_doi else 'LECH', tuple(y.shape)))
    # kiem tra kich thuoc tung khoi cua ban chinh, doi chieu bang 3.1
    m = InpaintAE(2, True)
    x = torch.zeros(2, 2, 512)
    for i, lop in enumerate(m.ma_hoa):
        x = lop(x)
        if isinstance(lop, nn.MaxPool1d):
            print('  sau khoi ma hoa: (T=%d, C=%d)' % (x.shape[2], x.shape[1]))
    h, _ = m.lstm(x.permute(0, 2, 1))
    print('  sau Bi-LSTM     : (T=%d, C=%d)' % (h.shape[1], h.shape[2]))
    x = h.permute(0, 2, 1)
    for lop in m.giai_ma:
        x = lop(x)
        if isinstance(lop, nn.Conv1d):
            print('  sau khoi giai ma: (T=%d, C=%d)' % (x.shape[2], x.shape[1]))
