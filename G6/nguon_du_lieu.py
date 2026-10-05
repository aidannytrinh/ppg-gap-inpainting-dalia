# -*- coding: utf-8 -*-
"""Ghi nguon, phien ban, ten tep va ma kiem tra tep cua du lieu, muc 2.1 de cuong:
"Luu nguon tai, phien ban, ten tep va ma kiem tra tep de co the tai lap."

Ket qua: results/nguon_du_lieu.json gom SHA-256 cua 15 tep SX.pkl goc, readme, va cac tep cache npz
ma chuong trinh thuc su doc.

Dung: python nguon_du_lieu.py
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
MOD = HERE.parent
GOC = MOD / 'data' / 'PPG_FieldStudy'
CACHE = MOD / 'dalia_cache'


def sha256(p, khoi=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(khoi), b''):
            h.update(b)
    return h.hexdigest()


def mo_ta(p):
    return dict(tep=p.relative_to(MOD).as_posix(), kich_thuoc_byte=p.stat().st_size, sha256=sha256(p))


if __name__ == '__main__':
    goc = [mo_ta(GOC / f'S{s}' / f'S{s}.pkl') for s in range(1, 16)]
    for g in goc:
        print(g['tep'], g['sha256'][:16])
    readme = GOC / 'PPG_FieldStudy_readme.pdf'
    cache = [mo_ta(p) for p in sorted(CACHE.glob('*.npz'))]
    kq = dict(
        bo_du_lieu='PPG-DaLiA (PPG_FieldStudy)',
        nguon='UCI Machine Learning Repository, DOI 10.24432/C53890',
        tac_gia='A. Reiss, I. Indlekofer, P. Schmidt, K. Van Laerhoven',
        phien_ban='Ban dong bo SX.pkl kem PPG_FieldStudy_readme.pdf (tep trong goi tai ve, khong ghi so phien ban rieng)',
        ngay_tinh_ma=datetime.now().isoformat(timespec='seconds'),
        readme=mo_ta(readme) if readme.exists() else None,
        tep_goc=goc,
        tep_cache_chuong_trinh_doc=cache,
        cach_tao_cache=('dalia_S*.npz va imu_S*.npz tao boi make_cache.py (BVP, ACC co tay, nhan hoat dong); '
                        'hr_S*.npz tao boi G6/prep_hr.py (nhan HR tu ECG va rpeaks). Khong xu ly them.'),
    )
    (HERE / 'results' / 'nguon_du_lieu.json').write_text(json.dumps(kq, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Da ghi results/nguon_du_lieu.json: {len(goc)} tep goc, {len(cache)} tep cache')
