# -*- coding: utf-8 -*-
"""Khao sat do nhay theo vi tri khoang thieu, muc 2.3 de cuong:
"Khao sat do nhay theo vi tri, neu thuc hien, dung bo vi tri dinh truoc {64, 128, 192, 256, 320},
ap dung giong nhau cho moi phuong phap va bao cao tach khoi ket qua chinh."

Chi danh gia, khong huan luyen, khong doi cau hinh: dung dung cac mo hinh trong khoa_cau_hinh.json
(da khoa truoc khi cham test) va bo doc HR chon tren S1-S12. Tap test S13-S15.
Voi moi vi tri a, khoang thieu la a..a+127, cung mot mat na cho moi phuong phap.
Vi tri a=192 trung voi giao thuc chinh nen phai tai lap dung so cua E1 (kiem tra cheo).

Moi vi tri ghi results/khaosat_vitri_a{a}.json (bo qua neu da co), sau do gop:
  python khao_sat_vi_tri.py --cho_phep_test            chay du 5 vi tri
  python khao_sat_vi_tri.py --cho_phep_test --a 64     chay mot vi tri
  python khao_sat_vi_tri.py --gop                      gop thanh results/khaosat_vitri.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

import baselines as B
import data as D
import evaluate as E

RES = Path(__file__).resolve().parent / 'results'
VI_TRI = [64, 128, 192, 256, 320]


def kiem_tra_dieu_kien(ds, a):
    """Muc 2.2 ap cho mat na moi: dem cua so khong huu han hoac sigma_O <= 1e-6 duoi mat na a..a+127."""
    M = D.mat_na(a)
    loai = {}
    for sid in ds:
        d = D.chi_muc(sid)
        idx = np.flatnonzero(d['du_dk'])
        R = np.stack([D.cua_so(d['bvp'], i) for i in idx]).astype(np.float64)
        Q = R[:, M == 0]
        sg = Q.std(axis=1)
        loai[sid] = int((~np.isfinite(R).all(axis=1) | (sg <= D.EPS_SIGMA)).sum())
    return loai


def chay_vi_tri(a):
    f = RES / f'khaosat_vitri_a{a}.json'
    if f.exists():
        print(f'[co san] {f.name}')
        return
    k = E.doc_khoa()
    pp = {t: E.pp_doi_chuan(t) for t in B.DOI_CHUAN}
    for nhan in ('mo_hinh_chinh', 'e2a_bo_lstm', 'e2b_lam0', 'e2c_them_acc'):
        pp[f'{nhan}[{k[nhan]}]'] = E.pp_mo_hinh(k[nhan])
    ds = D.SPLITS['test']
    loai = kiem_tra_dieu_kien(ds, a)
    print(f'Vi tri a={a}: khoang thieu {a}..{a + D.GAP - 1} | cua so khong du dieu kien: {loai}')
    bang, _ = E.chay(ds, pp, a=a, can_acc=True)
    kq = E.tong_hop(bang, list(pp))
    kq['ghep_cap_so_voi_lap_chu_ky'] = E.ghep_cap(bang, list(pp), 'lap_chu_ky')
    kq['vi_tri'] = dict(a=a, b=a + D.GAP - 1, mau_quan_sat_truoc=a, mau_quan_sat_sau=D.WIN - (a + D.GAP))
    kq['cua_so_khong_du_dieu_kien'] = loai
    f.write_text(json.dumps(kq, ensure_ascii=False, indent=2), encoding='utf-8')
    E.in_bang(kq, list(pp))
    print('Da ghi', f.name)


def gop():
    out = dict(quy_tac='muc 2.3: bo vi tri {64,128,192,256,320}, cung mat na cho moi phuong phap, test S13-S15, '
                       'mo hinh va bo doc HR da khoa; bao cao tach khoi ket qua chinh',
               vi_tri={})
    for a in VI_TRI:
        kq = json.loads((RES / f'khaosat_vitri_a{a}.json').read_text(encoding='utf-8'))
        ten_pp = [t for t in kq if isinstance(kq[t], dict) and 'rmse_macro' in kq[t]]
        out['vi_tri'][a] = dict(
            so_cua_so=kq['so_cua_so'], tap_hop_le_chung=kq['tap_hop_le_chung'],
            cua_so_khong_du_dieu_kien=kq['cua_so_khong_du_dieu_kien'],
            sai_so_nen_bo_doc=kq['sai_so_nen_bo_doc']['mae_macro'],
            phuong_phap={t: {m: kq[t][m] for m in ('mae_macro', 'rmse_macro', 'mae_hr_pres', 'mae_hr_ecg',
                                                   'ty_le_doc_duoc_hr', 'ty_le_du_phong')} for t in ten_pp},
            ghep_cap_so_voi_lap_chu_ky={t: {m: v[m] for m in ('trung_binh', 'so_doi_tuong_tot_hon', 'tong_doi_tuong')}
                                        for t, v in kq['ghep_cap_so_voi_lap_chu_ky'].items()})
    # kiem tra cheo: a=192 phai trung E1
    e1 = json.loads((RES / 'e1_test.json').read_text(encoding='utf-8'))
    lech = {t: abs(out['vi_tri'][192]['phuong_phap'][t]['mae_macro'] - e1[t]['mae_macro'])
            for t in out['vi_tri'][192]['phuong_phap']}
    out['kiem_tra_a192_so_voi_e1'] = dict(chenh_lech_mae_lon_nhat=max(lech.values()), khop=max(lech.values()) < 1e-3)
    (RES / 'khaosat_vitri.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Kiem tra a=192 so voi E1:', out['kiem_tra_a192_so_voi_e1'])
    for a in VI_TRI:
        r = out['vi_tri'][a]['phuong_phap']
        print('a=%3d  ' % a + '  '.join('%s %.4f' % (t[:14], r[t]['mae_macro']) for t in r))
    print('Da ghi results/khaosat_vitri.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cho_phep_test', action='store_true')
    ap.add_argument('--a', type=int, choices=VI_TRI)
    ap.add_argument('--gop', action='store_true')
    ap.add_argument('--luong', type=int, default=0, help='so luong torch moi tien trinh (0 = mac dinh)')
    x = ap.parse_args()
    if x.gop:
        gop()
        raise SystemExit(0)
    if not x.cho_phep_test:
        raise SystemExit('Tap test dang khoa. Chi chay sau khi da khoa cau hinh. Them --cho_phep_test.')
    if not E.KHOA.exists():
        raise SystemExit('Chua co khoa_cau_hinh.json')
    if x.luong:
        import torch
        torch.set_num_threads(x.luong)
    for a in ([x.a] if x.a else VI_TRI):
        chay_vi_tri(a)
