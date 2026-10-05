# -*- coding: utf-8 -*-
"""Bo doc nhip tim co dinh, dung chung cho moi phuong phap (muc 4.2 va 4.3 de cuong).

Bam theo de cuong da chinh sua:
  - Muc 4.2 : bo doc nhan cua so BVP 8 giay DA HOAN CHINH, cung bo loc va cung quy tac do dinh.
  - CT (12) : H(r) = 60 / trung binh cua Delta_t, voi Delta_t_j = (p_{j+1} - p_j)/64.
              Day la 60 chia cho khoang cach dinh trung binh, khong phai trung binh cua 60/Delta_t_j.
  - Muc 4.3 : butter(4, [0.5, 8], btype="bandpass", fs=64, output="sos") + sosfiltfilt tren tung cua so.
              N=4 la tham so thiet ke, khong phai bac tong sau hai luot.
  - Muc 4.3 : find_peaks co distance tinh bang MAU, khong phai giay.
              Gioi han HR va prominence chon tu train/validation, doi sang so mau truoc khi dung,
              giu co dinh khi test. Khong mac dinh 180 bpm phu hop moi doi tuong.
  - Muc 4.3 : can it nhat hai dinh hop le, nguoc lai cua so khong co HR.

Tieu chi chon tham so: de cuong khong chi dinh tieu chi, chi noi chon tu train/validation.
Muc 4.2 yeu cau luon bao cao H(PPG day du) so voi ECG de biet sai so nen cua bo doc, nen tieu chi
duoc dung o day la: toi thieu MAE giua H(PPG day du) va nhan HR tu ECG tren train + validation,
trong so cac cau hinh co ty le cua so doc duoc HR khong duoi NGUONG_HOP_LE.

Dung: python hr_reader.py --chon   -> do luoi tham so, ghi results/bodoc_hr.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

import data as D

RES = Path(__file__).resolve().parent / 'results'
CAU_HINH = RES / 'bodoc_hr.json'

LO, HI, BAC = 0.5, 8.0, 4          # dai thong va tham so thiet ke Butterworth
NGUONG_HOP_LE = 0.95               # ty le cua so toi thieu phai doc duoc HR
LUOI_HR_MAX = [150, 160, 170, 180, 190, 200]        # bpm, doi sang distance bang mau
LUOI_PROM = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]         # ty le so voi do lech chuan cua so da loc

_sos = None


def sos():
    global _sos
    if _sos is None:
        _sos = butter(BAC, [LO, HI], btype='bandpass', fs=D.FS, output='sos')
    return _sos


def loc(x):
    """Loc thong dai hai chieu tren tung cua so. x co dang (512,) hoac (N, 512)."""
    return sosfiltfilt(sos(), np.asarray(x, dtype=np.float64), axis=-1)


def hr_max_sang_mau(hr_max_bpm):
    """Doi tran nhip tim (bpm) sang khoang cach dinh toi thieu tinh bang MAU.

    Tran HR cang cao thi khoang cach toi thieu cang ngan: 60/HR_max giay, nhan FS mau moi giay.
    """
    return max(1, int(round(60.0 / float(hr_max_bpm) * D.FS)))


def hr_tu_cua_so(x_loc, distance, prom_ty_le):
    """HR cua mot cua so da loc, theo cong thuc (12). Tra ve NaN neu it hon hai dinh hop le."""
    prom = None if prom_ty_le <= 0 else prom_ty_le * float(np.std(x_loc))
    p, _ = find_peaks(x_loc, distance=distance, prominence=prom)
    if len(p) < 2:
        return np.nan
    dt = np.diff(p) / D.FS                      # Delta_t_j, don vi giay
    return 60.0 / float(np.mean(dt))


def doc_hr(R, ch=None):
    """HR cho mot lo cua so R dang (N, 512) o don vi goc. Tra ve mang (N,) co the chua NaN."""
    ch = ch or tai_cau_hinh()
    d, p = ch['distance_mau'], ch['prominence_ty_le']
    X = loc(np.atleast_2d(R))
    return np.array([hr_tu_cua_so(x, d, p) for x in X], dtype=np.float64)


def tai_cau_hinh():
    if not CAU_HINH.exists():
        raise FileNotFoundError(f'Chua chon tham so bo doc HR. Chay: python hr_reader.py --chon')
    return json.loads(CAU_HINH.read_text(encoding='utf-8'))


# ------------------------------------------------------------------ chon tham so
def _gom(ds):
    """Gom cua so du dieu kien cua cac doi tuong, tra ve (R da loc, nhan HR tu ECG)."""
    Xs, hrs = [], []
    for sid in ds:
        d = D.chi_muc(sid)
        idx = np.flatnonzero(d['du_dk'])
        R = np.stack([D.cua_so(d['bvp'], i) for i in idx])
        Xs.append(loc(R))
        hrs.append(d['hr'][idx])
        print(f'  S{sid}: {len(idx)} cua so')
    return np.concatenate(Xs), np.concatenate(hrs)


def chon_tham_so(ds=None, f_out=None):
    """Do luoi tren train + validation. Test khong tham gia buoc nay.

    Mac dinh: chia co dinh S1-S12, ghi results/bodoc_hr.json.
    Voi E5 (muc 2.4): truyen ds = train + val cua fold va f_out rieng cho fold.
    """
    RES.mkdir(exist_ok=True)
    ds = ds or (D.SPLITS['train'] + D.SPLITS['val'])
    f_out = f_out or CAU_HINH
    print(f'Nap va loc cua so cua S{ds} (khong dung test):')
    X, hr_ecg = _gom(ds)
    print(f'Tong {len(X)} cua so. Do luoi {len(LUOI_HR_MAX)}x{len(LUOI_PROM)} cau hinh.\n')

    bang = []
    for hm in LUOI_HR_MAX:
        d = hr_max_sang_mau(hm)
        for pr in LUOI_PROM:
            hr = np.array([hr_tu_cua_so(x, d, pr) for x in X])
            ok = np.isfinite(hr)
            ty_le = float(ok.mean())
            mae = float(np.mean(np.abs(hr[ok] - hr_ecg[ok]))) if ok.any() else np.inf
            bang.append(dict(hr_max=hm, distance_mau=d, prominence_ty_le=pr,
                             ty_le_hop_le=round(ty_le, 4), mae_nen=round(mae, 3)))
            print('  HR_max=%3d (distance=%2d mau)  prom=%.2f  ty_le=%.4f  MAE_nen=%6.3f bpm'
                  % (hm, d, pr, ty_le, mae))

    hop_le = [b for b in bang if b['ty_le_hop_le'] >= NGUONG_HOP_LE]
    if not hop_le:
        raise RuntimeError('Khong cau hinh nao dat nguong ty le hop le')
    tot = min(hop_le, key=lambda b: b['mae_nen'])
    ket = dict(tot, nguon='train + validation', doi_tuong=list(ds), so_cua_so=len(X),
               nguong_hop_le=NGUONG_HOP_LE, dai_thong=[LO, HI], bac_thiet_ke=BAC,
               tieu_chi='toi thieu MAE giua H(PPG day du) va nhan HR tu ECG',
               ghi_chu='prominence tinh theo ty le voi do lech chuan cua so da loc, nen khong doi khi doi don vi bien do',
               bang=bang)
    f_out.write_text(json.dumps(ket, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nChon: HR_max=%d bpm -> distance=%d mau, prominence=%.2f x do lech chuan'
          % (tot['hr_max'], tot['distance_mau'], tot['prominence_ty_le']))
    print('Sai so nen cua bo doc tren train+val: MAE=%.3f bpm, ty le doc duoc %.2f%%'
          % (tot['mae_nen'], tot['ty_le_hop_le'] * 100))
    print('Da ghi:', f_out)
    return ket


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--chon', action='store_true', help='do luoi va chot tham so bo doc HR')
    a = ap.parse_args()
    if a.chon:
        chon_tham_so()
    else:
        d = D.chi_muc(1)
        R = np.stack([D.cua_so(d['bvp'], i) for i in range(200, 210)])
        X = loc(R)
        for hm in (180, 200):
            dd = hr_max_sang_mau(hm)
            hr = np.array([hr_tu_cua_so(x, dd, 0.1) for x in X])
            print('HR_max=%d -> distance=%d mau | HR doc duoc:' % (hm, dd),
                  np.round(hr, 1), '| ECG:', np.round(d['hr'][200:210], 1))
