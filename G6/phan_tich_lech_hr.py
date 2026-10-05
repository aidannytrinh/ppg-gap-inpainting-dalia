# -*- coding: utf-8 -*-
"""Chieu lech HR cua tung phuong phap so voi HR doc tu PPG day du tren test E1 (bo sung cho muc 4.2).
Chi doc results/e1_test_theo_cuaso.csv, khong tinh lai du doan.

Voi moi phuong phap: lech trung binh H(r_out) - H(r), ty le cua so thap hon / cao hon qua 0,5 bpm.
Dung: python phan_tich_lech_hr.py   -> results/e1_lech_hr.json
"""
import csv
import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent / 'results'

if __name__ == '__main__':
    with open(RES / 'e1_test_theo_cuaso.csv', encoding='utf-8-sig') as fh:
        rows = list(csv.DictReader(fh))
    day_du = np.array([float(r['hr_day_du']) for r in rows])
    ecg = np.array([float(r['hr_ecg']) for r in rows])
    ten_pp = [k[3:] for k in rows[0] if k.startswith('hr_') and k not in ('hr_ecg', 'hr_day_du')]
    kq = dict(tap='test S13-S15 (E1)', so_cua_so=len(rows), nguong_bpm=0.5,
              day_du_tru_ecg_trung_binh=round(float((day_du - ecg).mean()), 3), phuong_phap={})
    for t in ten_pp:
        d = np.array([float(r[f'hr_{t}']) for r in rows]) - day_du
        kq['phuong_phap'][t] = dict(lech_trung_binh=round(float(d.mean()), 3),
                                    ty_le_thap_hon=round(float((d < -0.5).mean()), 4),
                                    ty_le_cao_hon=round(float((d > 0.5).mean()), 4))
        print('%-32s lech TB %+7.2f | thap hon %.3f | cao hon %.3f' % (t, d.mean(), (d < -0.5).mean(), (d > 0.5).mean()))
    (RES / 'e1_lech_hr.json').write_text(json.dumps(kq, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Da ghi results/e1_lech_hr.json')
