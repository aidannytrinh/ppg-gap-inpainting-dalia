# -*- coding: utf-8 -*-
"""Trich nhan HR tham chieu tu ECG (truong 'label' trong .pkl goc) ra cache npz.

PPG-DaLiA cong bo 'label' la HR tinh tu ECG tren cua so 8 giay, buoc 2 giay.
Cua so cua de tai cung la 512 mau (8 s) buoc 128 mau (2 s) o 64 Hz, nen label[i]
phai ung dung voi cua so i = bvp[i*128 : i*128+512]. Script kiem tra dieu do
truoc khi ghi, va bao cao do lech neu co.

Dung: python prep_hr.py            -> chay het 15 doi tuong
      python prep_hr.py --probe    -> chi S1, de kiem tra truoc
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

MOD = Path(__file__).resolve().parent.parent
PKL_DIR = MOD / 'data' / 'PPG_FieldStudy'
CACHE = MOD / 'dalia_cache'
WIN, STEP = 512, 128          # 8 s va 2 s o 64 Hz


def so_cua_so(n):
    """Cong thuc (2) trong de cuong: so cua so truoc buoc loai cua so khong hop le."""
    return max(0, (n - WIN) // STEP + 1)


def xu_ly(sid, ghi=True):
    f = PKL_DIR / f'S{sid}' / f'S{sid}.pkl'
    t0 = time.time()
    with open(f, 'rb') as fh:
        d = pickle.load(fh, encoding='latin1')

    bvp = np.asarray(d['signal']['wrist']['BVP'], dtype=np.float32).ravel()
    hr = np.asarray(d['label'], dtype=np.float32).ravel()
    act = np.asarray(d['activity']).ravel()
    rp = np.asarray(d['rpeaks']).ravel()

    n = len(bvp)
    ky_vong = so_cua_so(n)
    lech = len(hr) - ky_vong

    # doi chieu voi cache da co, de chac chan cung mot phien ban du lieu
    cu = np.load(CACHE / f'dalia_S{sid}.npz')
    khop_bvp = len(cu['bvp']) == n and np.allclose(cu['bvp'], bvp, atol=1e-4)
    khop_act = len(cu['act']) == len(act) and np.array_equal(cu['act'].astype(int), act.astype(int))

    hop_le = np.isfinite(hr)
    print(f'S{sid:<3} bvp={n:>7}  cua_so_ky_vong={ky_vong:>5}  len(label)={len(hr):>5}  '
          f'lech={lech:>+3}  HR[{np.nanmin(hr):5.1f},{np.nanmax(hr):6.1f}] '
          f'NaN={np.sum(~hop_le):>3}  rpeaks={len(rp):>6}  '
          f'khop_cache: bvp={khop_bvp} act={khop_act}  ({time.time()-t0:.0f}s)')

    if ghi:
        np.savez_compressed(CACHE / f'hr_S{sid}.npz', hr=hr, rpeaks=rp.astype(np.int64))
    return dict(sid=sid, n=n, ky_vong=ky_vong, n_hr=len(hr), lech=lech,
                khop_bvp=bool(khop_bvp), khop_act=bool(khop_act), nan=int(np.sum(~hop_le)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', action='store_true', help='chi chay S1, khong ghi file')
    a = ap.parse_args()
    if not PKL_DIR.exists():
        sys.exit(f'Khong thay bo goc: {PKL_DIR}')

    ds = [1] if a.probe else list(range(1, 16))
    kq = [xu_ly(s, ghi=not a.probe) for s in ds]

    print('\n--- TONG KET ---')
    print('Tong so cua so ky vong (truoc buoc loai):', sum(r['ky_vong'] for r in kq))
    xau = [r for r in kq if r['lech'] != 0 or not r['khop_bvp'] or not r['khop_act']]
    print('Doi tuong co van de:', [r['sid'] for r in xau] if xau else 'khong co')
