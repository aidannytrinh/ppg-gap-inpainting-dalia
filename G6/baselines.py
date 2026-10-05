# -*- coding: utf-8 -*-
"""Ba phuong phap doi chuan, muc 3.4 de cuong. Khong phuong phap nao duoc doc phan bi che.

  1. Noi suy tuyen tinh : noi hai mau quan sat sat khoang thieu.
  2. Spline bac ba      : CubicSpline tren 64 mau quan sat ngay truoc va 64 mau ngay sau,
                          moc thoi gian that, bc_type="not-a-knot", khong ngoai suy.
                          Khong khop spline bang gia tri that trong khoang thieu.
  3. Lap chu ky gan nhat: hai dinh hop le gan nhat TRUOC khoang thieu, lay chu ky giua chung,
                          keo dai tuan hoan voi pha tinh tu vi tri dinh cuoi, khong khoi dong lai pha.
                          Tron tuyen tinh 8 mau phia trong moi mep de noi voi mau quan sat,
                          khong sua phan da biet. Thieu chu ky day du hoac do dinh that bai thi
                          dung noi suy tuyen tinh va ghi nhan de bao cao ty le.

Moi ham nhan (z_in, M) chu khong nhan z day du, nen ve mat cau truc khong the doc vung bi che.
z_in la tin hieu da chuan hoa voi mau thieu dat bang 0, dung cong thuc (1).
Cac phep tren deu dong bien voi phep affine nen lam trong mien z cho ket qua giong het lam
tren r roi chuan hoa, dung nguyen tac "cung quy tac chuan hoa" o muc 3.4.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks

import data as D
import hr_reader as H

N_MOC_SPLINE = 64        # so mau quan sat moi ben dung de khop spline
N_TRON = 8               # so mau tron tuyen tinh phia trong moi mep


def vi_tri_gap(M):
    """Tra ve (a, b): chi so dau va cuoi cua khoang thieu."""
    idx = np.flatnonzero(M == 1)
    a, b = int(idx[0]), int(idx[-1])
    if b - a + 1 != len(idx):
        raise ValueError('Khoang thieu khong lien tuc')
    return a, b


def noi_suy_tuyen_tinh(z_in, M):
    """Doi chuan 1. Noi thang tu mau quan sat ngay truoc toi mau ngay sau khoang thieu."""
    a, b = vi_tri_gap(M)
    t = np.arange(a, b + 1, dtype=np.float64)
    out = np.array(z_in, dtype=np.float64, copy=True)
    out[a:b + 1] = np.interp(t, [a - 1, b + 1], [z_in[a - 1], z_in[b + 1]])
    return out


def spline_bac_ba(z_in, M):
    """Doi chuan 2. CubicSpline not-a-knot tren 64 mau quan sat moi ben, khong ngoai suy."""
    a, b = vi_tri_gap(M)
    lo, hi = a - N_MOC_SPLINE, b + N_MOC_SPLINE
    if lo < 0 or hi > len(z_in) - 1:
        raise ValueError(f'Khong du {N_MOC_SPLINE} mau moi ben cho spline (a={a}, b={b})')
    moc = np.concatenate([np.arange(lo, a), np.arange(b + 1, hi + 1)]).astype(np.float64)
    gia_tri = np.concatenate([z_in[lo:a], z_in[b + 1:hi + 1]]).astype(np.float64)
    cs = CubicSpline(moc, gia_tri, bc_type='not-a-knot', extrapolate=False)
    out = np.array(z_in, dtype=np.float64, copy=True)
    out[a:b + 1] = cs(np.arange(a, b + 1, dtype=np.float64))
    return out


def lap_chu_ky(z_in, M, ch=None):
    """Doi chuan 3. Tra ve (tin hieu, da_dung_du_phong).

    Dinh duoc do tren phan quan sat TRUOC khoang thieu, sau khi loc thong dai, dung dung
    quy tac do dinh cua bo doc HR (muc 4.3). Mau dung de lap lay tu tin hieu chua loc,
    vi dich tai tao nam o mien z chua loc.
    """
    ch = ch or H.tai_cau_hinh()
    a, b = vi_tri_gap(M)
    truoc = np.asarray(z_in[:a], dtype=np.float64)

    dinh = np.array([], dtype=int)
    if len(truoc) > 3 * len(H.sos()) * 2:                  # du dai de loc hai chieu
        x = H.loc(truoc)
        prom = None if ch['prominence_ty_le'] <= 0 else ch['prominence_ty_le'] * float(np.std(x))
        dinh, _ = find_peaks(x, distance=ch['distance_mau'], prominence=prom)

    if len(dinh) < 2:
        return noi_suy_tuyen_tinh(z_in, M), True           # phuong an du phong

    p1, p2 = int(dinh[-2]), int(dinh[-1])
    T = p2 - p1
    if T < 2 or p1 < 0:
        return noi_suy_tuyen_tinh(z_in, M), True
    mau = np.asarray(z_in[p1:p2], dtype=np.float64)        # mot chu ky tron ven, bat dau tai mot dinh

    def gia_tri_tai(t):
        """Pha tinh tu dinh cuoi p2: t = p2 ung voi dau chu ky."""
        return mau[(t - p2) % T]

    out = np.array(z_in, dtype=np.float64, copy=True)
    out[a:b + 1] = [gia_tri_tai(t) for t in range(a, b + 1)]

    # tron tuyen tinh phia trong hai mep, chi sua trong khoang thieu
    d_trai = float(z_in[a - 1] - gia_tri_tai(a - 1))
    # Khi khoang thieu cham cuoi cua so (chi xay ra trong phan cai tien do tre, a = 384) thi khong co mau
    # phia sau de noi, bo tron mep phai. Voi moi a <= 320 cua giao thuc chinh nhanh nay khong bao gio chay.
    d_phai = float(z_in[b + 1] - gia_tri_tai(b + 1)) if b + 1 < len(z_in) else 0.0
    n = min(N_TRON, (b - a + 1) // 2)
    for k in range(n):
        out[a + k] += d_trai * (1.0 - k / n)
        out[b - k] += d_phai * (1.0 - k / n)
    return out, False


DOI_CHUAN = {
    'tuyen_tinh': lambda z_in, M, ch=None: (noi_suy_tuyen_tinh(z_in, M), False),
    'spline': lambda z_in, M, ch=None: (spline_bac_ba(z_in, M), False),
    'lap_chu_ky': lap_chu_ky,
}


if __name__ == '__main__':
    d = D.chi_muc(1)
    M = D.mat_na(D.A_EVAL)
    ch = H.tai_cau_hinh() if H.CAU_HINH.exists() else dict(distance_mau=21, prominence_ty_le=0.1)
    n_dp = 0
    sai = {k: [] for k in DOI_CHUAN}
    for i in range(300, 400):
        r = D.cua_so(d['bvp'], i)
        z, z_in, mu, s, sg = D.chuan_hoa(r, M)
        for ten, f in DOI_CHUAN.items():
            y, dp = f(z_in, M, ch)
            n_dp += int(dp and ten == 'lap_chu_ky')
            sai[ten].append(np.mean(np.abs(y[M == 1] - z[M == 1])))
            assert np.allclose(y[M == 0], z_in[M == 0]), f'{ten} da sua phan da biet'
    for ten in DOI_CHUAN:
        print('%-12s MAE tren doan thieu = %.4f' % (ten, np.mean(sai[ten])))
    print('So lan lap chu ky phai dung du phong: %d/100' % n_dp)
