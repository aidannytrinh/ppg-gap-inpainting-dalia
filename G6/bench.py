# -*- coding: utf-8 -*-
"""Danh gia tai nguyen va do tre, muc 4.5 de cuong.

  - batch = 1, che do eval va inference_mode; ghi CPU, so luong, phien ban runtime, kieu so.
  - Khoi dong 100 lan roi do 1.000 lan; bao cao trung vi, phan vi 95% va trung binh.
  - Tach thoi gian rieng mo hinh voi toan pipeline (chuan hoa + mo hinh + ghep + ve don vi goc).
  - Bao cao so tham so, kich thuoc trong so, RAM cuc dai cua tien trinh neu cong cu cho phep.
  - Do tre cho ngu canh o muc 1.2 cong bo rieng, tinh tu giao thuc, khong gop vao thoi gian tinh.
  - Do tren may tinh, khong thay the thu nghiem tren thiet bi deo.

CHI CHAY KHI MAY RANH (khong co huan luyen song song), neu khong so do bi sai.
Dung: python bench.py
"""
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

import data as D
import train as T
from model import InpaintAE, dem_tham_so

HERE = Path(__file__).resolve().parent
RES, W = HERE / 'results', HERE / 'weights'
N_KHOI_DONG, N_DO = 100, 1000


def thong_ke(ms):
    ms = sorted(ms)
    return dict(trung_vi_ms=round(statistics.median(ms), 4),
                p95_ms=round(ms[int(0.95 * len(ms)) - 1], 4),
                trung_binh_ms=round(statistics.fmean(ms), 4))


def ten_cpu():
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
        return winreg.QueryValueEx(k, 'ProcessorNameString')[0].strip()
    except Exception:
        return platform.processor()


def ram_cuc_dai_mb():
    try:
        import psutil
        mi = psutil.Process(os.getpid()).memory_info()
        return round(getattr(mi, 'peak_wset', mi.rss) / 2 ** 20, 1)
    except Exception:
        return None


def do(fn):
    for _ in range(N_KHOI_DONG):
        fn()
    t = []
    for _ in range(N_DO):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000)
    return thong_ke(t)


def main():
    khoa = json.loads((RES / 'khoa_cau_hinh.json').read_text(encoding='utf-8'))
    id_ = khoa['mo_hinh_chinh']
    m = InpaintAE(2, dung_lstm=True)
    m.load_state_dict(torch.load(W / f'{id_}.pt', map_location='cpu'))
    m.eval()

    d = D.nap_doi_tuong(13)                                # mot cua so test bat ky lam dau vao mau
    r_np = np.array(D.cua_so(d['bvp'], 1000), dtype=np.float32)
    R = torch.from_numpy(r_np).unsqueeze(0)                # (1, 512)
    a = torch.tensor([D.A_EVAL])
    u, z, M = T.lam_dau_vao(R, a)

    with torch.inference_mode():
        def chi_mo_hinh():
            m(u)

        def toan_pipeline():
            u2, z2, M2 = T.lam_dau_vao(R, a)               # chuan hoa theo phan quan sat, CT (3)(4)
            zh = m(u2)
            z_out = (1 - M2) * z2 + M2 * zh                # CT (5)
            O = 1 - M2
            mu = (R * O).sum(1, keepdim=True) / O.sum(1, keepdim=True)
            s = torch.clamp(torch.sqrt((((R - mu) ** 2) * O).sum(1, keepdim=True) / O.sum(1, keepdim=True)),
                            min=D.EPS_SIGMA)
            return (1 - M2) * R + M2 * (mu + s * zh)       # CT (6)

        t_mh = do(chi_mo_hinh)
        t_pl = do(toan_pipeline)

    b = D.A_EVAL + D.GAP - 1
    sau_gap = (D.WIN - 1 - b) / D.FS
    kq = dict(
        mo_hinh=id_,
        moi_truong=dict(cpu=ten_cpu(), so_luong_torch=torch.get_num_threads(), so_nhan_logic=os.cpu_count(),
                        he_dieu_hanh=platform.platform(), python=sys.version.split()[0], torch=torch.__version__,
                        kieu_so=str(next(m.parameters()).dtype), batch=1, che_do='eval + inference_mode',
                        khoi_dong=N_KHOI_DONG, so_lan_do=N_DO),
        thoi_gian_chi_mo_hinh=t_mh,
        thoi_gian_toan_pipeline=t_pl,
        tham_so=dem_tham_so(m),
        kich_thuoc_trong_so_kb=round((W / f'{id_}.pt').stat().st_size / 1024, 1),
        ram_cuc_dai_tien_trinh_mb=ram_cuc_dai_mb(),
        do_tre_cho_ngu_canh=dict(
            ghi_chu='tinh tu giao thuc khoang thieu co dinh 192..319, khong phai thoi gian tinh',
            giay_cho_sau_khi_het_khoang_mat=round(sau_gap, 3),
            giay_tu_luc_bat_dau_mat=round(sau_gap + D.GAP / D.FS, 3)),
        gioi_han='Do tren may tinh ca nhan, khong thay the thu nghiem tren thiet bi deo.')
    (RES / 'bench.json').write_text(json.dumps(kq, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(kq, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
