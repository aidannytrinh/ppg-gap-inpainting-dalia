# -*- coding: utf-8 -*-
"""Tao cache nho tu bo PPG-DaLiA goc de chay lai tu dau (muc 5.2: chay lai duoc tu README).

Logic giong het ../make_cache.py da dung de tao cache hien co, chi doi duong dan sang data/PPG_FieldStudy.
Moi SX.pkl ~1,4 GB nhung phan lon la tin hieu nguc 700 Hz khong dung; chi giu:
  dalia_SX.npz: bvp (BVP co tay 64 Hz, float32), act (nhan hoat dong 4 Hz, int8)
  imu_SX.npz  : acc_w (ACC co tay 32 Hz, 3 truc, float32)
Nhan HR tu ECG tao rieng bang prep_hr.py.

Dung: python tao_cache.py [--ra <thu muc>]
"""
import argparse
import pickle
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MOD = HERE.parent
PKL_DIR = MOD / 'data' / 'PPG_FieldStudy'


def build(sid, ra):
    with open(PKL_DIR / f'S{sid}' / f'S{sid}.pkl', 'rb') as f:
        d = pickle.load(f, encoding='latin-1')
    bvp = np.asarray(d['signal']['wrist']['BVP']).flatten().astype(np.float32)
    acc = np.asarray(d['signal']['wrist']['ACC']).astype(np.float32)
    act = np.asarray(d['activity']).flatten().astype(np.int8)
    np.savez_compressed(ra / f'dalia_S{sid}.npz', bvp=bvp, act=act)
    np.savez_compressed(ra / f'imu_S{sid}.npz', acc_w=acc)
    return len(bvp), len(acc), len(act)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ra', default=str(MOD / 'dalia_cache'))
    ap.add_argument('--doi_tuong', type=int, nargs='+', default=list(range(1, 16)))
    a = ap.parse_args()
    ra = Path(a.ra)
    ra.mkdir(parents=True, exist_ok=True)
    for sid in a.doi_tuong:
        t0 = time.time()
        n1, n2, n3 = build(sid, ra)
        print(f'S{sid:<3} bvp {n1:>8} | acc {n2:>7} x3 | act {n3:>6} | {time.time() - t0:.1f} s')
