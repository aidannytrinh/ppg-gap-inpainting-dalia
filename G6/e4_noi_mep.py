# -*- coding: utf-8 -*-
"""E4 phan con lai: chat luong noi mep theo lambda, do tren VALIDATION S11-S12 (muc 4.4 de cuong:
"Kiem tra thay doi sai so dang song va chat luong noi mep; khong chon lambda bang test").

Khong huan luyen lai, chi dung trong so E4 da co. Test khong tham gia.
Voi moi cua so va khoang thieu co dinh 192..319 (a=192, b=319):
  - L_bnd (CT 8, 9): trung binh tren B = {a-1, a, b-1, b} cua
        [(z_out[t+1] - z_out[t]) - (z[t+1] - z[t])]^2
    z_out la tin hieu da ghep theo CT (5): ngoai khoang thieu la z quan sat, trong khoang thieu la du doan.
  - L_noi: cung bieu thuc nhung chi tren hai cap noi voi vung quan sat, t in {a-1, b}.
  - MAE vung thieu (CT 10) de doi chieu voi mae_val_tot_nhat da ghi khi huan luyen.
Gop macro theo CT (11). Ba doi chuan duoc tinh cung cach de lam moc tham chieu.

Dung: python e4_noi_mep.py   -> results/e4_noi_mep.json
"""
import json
from pathlib import Path

import numpy as np

import data as D
import evaluate as E
import hr_reader as H

RES = Path(__file__).resolve().parent / 'results'
E4 = ['e4_lam0', 'e4_lam0.1', 'e4_lam0.5', 'e4_lam1']


def chi_so_mep(Z, Y, M):
    """Z: dich chuan hoa (N,512). Y: du doan day du 512 mau cua phuong phap. Tra ve (L_bnd, L_noi) moi cua so."""
    a, b = int(np.flatnonzero(M == 1)[0]), int(np.flatnonzero(M == 1)[-1])
    Zo = np.where(M == 1, Y, Z)                      # CT (5): ghep theo mat na
    dZo, dZ = np.diff(Zo, axis=1), np.diff(Z, axis=1)
    e = (dZo - dZ) ** 2                              # e[:, t] ung voi cap (t, t+1)
    l_bnd = e[:, [a - 1, a, b - 1, b]].mean(axis=1)
    l_noi = e[:, [a - 1, b]].mean(axis=1)
    return l_bnd, l_noi


if __name__ == '__main__':
    ch = H.tai_cau_hinh()
    M = D.mat_na(D.A_EVAL)
    pp = {t: E.pp_doi_chuan(t) for t in ('tuyen_tinh', 'spline', 'lap_chu_ky')}
    for id_ in E4:
        pp[id_] = E.pp_mo_hinh(id_)

    cot = {t: dict(mae=[], bnd=[], noi=[]) for t in pp}
    sids = []
    for sid in D.SPLITS['val']:
        d = D.chi_muc(sid)
        idx = np.flatnonzero(d['du_dk'])
        R = np.stack([D.cua_so(d['bvp'], i) for i in idx]).astype(np.float64)
        Z, Z_in, mu, s = E.chuan_hoa_lo(R, M)
        sids += [sid] * len(idx)
        for t, f in pp.items():
            Y, _ = f(R, Z_in, M, None, ch)
            mae, _ = E.chi_so_gap(Z, Y, M)
            bnd, noi = chi_so_mep(Z, Y, M)
            cot[t]['mae'] += mae.tolist()
            cot[t]['bnd'] += bnd.tolist()
            cot[t]['noi'] += noi.tolist()
        print(f'  S{sid}: {len(idx)} cua so')

    sids = np.asarray(sids)
    kq = dict(tap='validation', doi_tuong=D.SPLITS['val'], so_cua_so=int(len(sids)),
              khoang_thieu=[D.A_EVAL, D.A_EVAL + D.GAP - 1],
              ghi_chu='L_bnd theo CT (8) tren 4 cap; L_noi chi 2 cap noi voi vung quan sat (a-1, b). Gop macro CT (11).',
              ket_qua={})
    print('\n%-12s %8s %10s %10s' % ('phuong phap', 'MAE', 'L_bnd', 'L_noi'))
    for t in pp:
        r = {}
        for k in ('mae', 'bnd', 'noi'):
            m, ng = E.gop_macro(cot[t][k], sids)
            r[k] = dict(macro=round(m, 5), theo_doi_tuong={s: round(v, 5) for s, v in ng.items()})
        kq['ket_qua'][t] = r
        print('%-12s %8.4f %10.5f %10.5f' % (t, r['mae']['macro'], r['bnd']['macro'], r['noi']['macro']))
    (RES / 'e4_noi_mep.json').write_text(json.dumps(kq, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Da ghi results/e4_noi_mep.json')
