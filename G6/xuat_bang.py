# -*- coding: utf-8 -*-
"""Xuat bang CSV va nhat ky that bai tu tep ket qua, muc 5.2 de cuong:
"bang CSV chi so theo cua so/doi tuong/hoat dong; ... nhat ky cac truong hop that bai".

  - bang_<ten>_theo_doituong.csv : MAE, RMSE, MAE_HR,pres, MAE_HR,ECG cua moi phuong phap theo doi tuong.
  - bang_<ten>_theo_hoatdong.csv : cung chi so theo tung hoat dong (chi co voi E1, tu E3).
  - nhatky_thatbai_<ten>.csv     : moi cua so ma bo doc HR khong tra duoc HR (voi bat ky phuong phap nao,
                                    ke ca PPG day du) hoac nhan ECG khong hop le, hoac lap chu ky phai dung
                                    phuong an du phong. Ghi ro ly do.
  - nhatky_thatbai_tomtat.json   : dem so truong hop theo ly do va phuong phap.
Bang theo cua so da co san: <ten>_theo_cuaso.csv (co sid, chi so cua so, giay bat dau).

Dung: python xuat_bang.py
"""
import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
CHI_SO = ('mae_macro', 'rmse_macro', 'mae_hr_pres', 'mae_hr_ecg')


def ghi_csv(p, hang):
    if not hang:
        return
    with open(p, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(hang[0]))
        w.writeheader()
        w.writerows(hang)
    print('Da ghi', p.name, f'({len(hang)} dong)')


def ten_pp(kq):
    return [k for k, v in kq.items() if isinstance(v, dict) and 'mae_theo_doi_tuong' in v]


def theo_doi_tuong(ten, kq):
    pp = ten_pp(kq)
    ds = sorted({int(s) for p in pp for s in kq[p]['mae_theo_doi_tuong']})
    hang = []
    for s in ds:
        r = dict(doi_tuong=f'S{s}')
        for p in pp:
            r[f'{p}|mae'] = kq[p]['mae_theo_doi_tuong'].get(str(s))
            r[f'{p}|rmse'] = kq[p]['rmse_theo_doi_tuong'].get(str(s))
            r[f'{p}|mae_hr_pres'] = kq[p]['mae_hr_pres_theo_doi_tuong'].get(str(s))
            if 'mae_hr_ecg_theo_doi_tuong' in kq[p]:
                r[f'{p}|mae_hr_ecg'] = kq[p]['mae_hr_ecg_theo_doi_tuong'].get(str(s))
        hang.append(r)
    r = dict(doi_tuong='gop deu (macro)')
    for p in pp:
        r[f'{p}|mae'], r[f'{p}|rmse'], r[f'{p}|mae_hr_pres'] = (kq[p]['mae_macro'], kq[p]['rmse_macro'],
                                                               kq[p]['mae_hr_pres'])
        if 'mae_hr_ecg_theo_doi_tuong' in kq[p]:
            r[f'{p}|mae_hr_ecg'] = kq[p]['mae_hr_ecg']
    hang.append(r)
    ghi_csv(RES / f'bang_{ten}_theo_doituong.csv', hang)


def theo_hoat_dong(ten, kq):
    if 'e3_theo_hoat_dong' not in kq:
        return
    hang = []
    for nhom, r0 in kq['e3_theo_hoat_dong'].items():
        r = dict(hoat_dong=nhom, so_cua_so=r0['so_cua_so'], so_doi_tuong=r0['so_doi_tuong'],
                 sai_so_nen_bo_doc=r0['sai_so_nen_bo_doc']['mae_macro'])
        for p, v in r0.items():
            if isinstance(v, dict) and 'rmse_macro' in v:          # chi cac phuong phap, bo muc sai so nen
                for m in CHI_SO:
                    r[f'{p}|{m}'] = v[m]
        hang.append(r)
    ghi_csv(RES / f'bang_{ten}_theo_hoatdong.csv', hang)


def nhat_ky_that_bai(ten):
    f = RES / f'{ten}_theo_cuaso.csv'
    with open(f, encoding='utf-8-sig') as fh:
        rows = list(csv.DictReader(fh))
    cot_hr = [c for c in rows[0] if c.startswith('hr_')]          # hr_ecg, hr_day_du, hr_<phuong phap>
    cot_dp = [c for c in rows[0] if c.startswith('duphong_')]
    hang, dem = [], {}
    for r in rows:
        ly_do = []
        for c in cot_hr:
            v = float(r[c])
            if not math.isfinite(v):
                ly_do.append('ECG khong hop le' if c == 'hr_ecg' else f'bo doc khong ra HR: {c[3:]}')
        for c in cot_dp:
            if r[c] == '1':
                ly_do.append(f'dung du phong: {c[8:]}')
        for l in ly_do:
            dem[l] = dem.get(l, 0) + 1
            hang.append(dict(sid=r['sid'], cua_so=r['win'], giay_bat_dau=r.get('giay_bat_dau', ''),
                             hoat_dong=r['act'], ly_do=l))
    p = RES / f'nhatky_thatbai_{ten}.csv'
    if hang:
        ghi_csv(p, hang)
    else:
        p.write_text('sid,cua_so,giay_bat_dau,hoat_dong,ly_do\n', encoding='utf-8-sig')
        print('Da ghi', p.name, '(khong co truong hop that bai nao)')
    return dict(so_cua_so=len(rows), so_truong_hop=len(hang), theo_ly_do=dem)


if __name__ == '__main__':
    tom_tat = {}
    ten_ds = ['doichuan_val', 'e1_test'] + [f'e5_f{k}_test' for k in range(1, 6)]
    for ten in ten_ds:
        if not (RES / f'{ten}.json').exists():
            print('Bo qua (chua co):', ten)
            continue
        kq = json.loads((RES / f'{ten}.json').read_text(encoding='utf-8'))
        theo_doi_tuong(ten, kq)
        theo_hoat_dong(ten, kq)
        tom_tat[ten] = nhat_ky_that_bai(ten)
    (RES / 'nhatky_thatbai_tomtat.json').write_text(json.dumps(tom_tat, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(tom_tat, ensure_ascii=False, indent=2))
