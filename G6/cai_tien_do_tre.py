# -*- coding: utf-8 -*-
"""CAI TIEN (ngoai yeu cau de cuong, bao cao tach rieng): danh doi giua do tre cho ngu canh va sai so.

Muc 1.2 de cuong: voi khoang thieu o giua cua so (192..319) he thong phai cho 3 giay sau khi doan mat ket thuc.
Khi trien khai, co the dat cua so sao cho doan mat nam muon hon de giam thoi gian cho:
    thoi gian cho sau doan mat = (512 - (a + 128)) / 64 giay
    a = 64 -> 5 s, 128 -> 4 s, 192 -> 3 s, 256 -> 2 s, 320 -> 1 s, 352 -> 0,5 s, 384 -> 0 s.
a <= 320 nam trong vung mo hinh duoc huan luyen (a ngau nhien 64..320); a = 352, 384 nam NGOAI vung huan luyen.

Do tren validation (S11, S12) de chon muc cho khuyen nghi va tren test (S13-S15) de bao cao. Khong huan luyen lai,
khong doi cau hinh da khoa. Phuong phap nao khong xac dinh duoc o mot vi tri thi khong chay:
  - spline can 64 mau quan sat moi phia -> chi a <= 320
  - noi suy tuyen tinh can 1 mau sau khoang thieu -> a <= 383
  - lap chu ky chi dung phia truoc; o a = 384 bo buoc tron mep phai (khong co mau de noi)

Dung:
  python cai_tien_do_tre.py --tap val
  python cai_tien_do_tre.py --tap test --cho_phep_test
  python cai_tien_do_tre.py --tap test --cho_phep_test --mo_hinh_them <id>     (them mot mo hinh, vd ban do tre thap)
  python cai_tien_do_tre.py --gop
"""
import argparse
import json
from pathlib import Path

import baselines as B
import data as D
import evaluate as E

RES = Path(__file__).resolve().parent / 'results'
VI_TRI = [64, 128, 192, 256, 320, 352, 384]


def cho(a):
    return (D.WIN - (a + D.GAP)) / D.FS


def phuong_phap(a, them):
    pp = {'lap_chu_ky': E.pp_doi_chuan('lap_chu_ky')}
    if a + D.GAP <= D.WIN - 1:
        pp['tuyen_tinh'] = E.pp_doi_chuan('tuyen_tinh')
    if a - B.N_MOC_SPLINE >= 0 and a + D.GAP - 1 + B.N_MOC_SPLINE <= D.WIN - 1:
        pp['spline'] = E.pp_doi_chuan('spline')
    k = E.doc_khoa()
    pp[f'mo_hinh_chinh[{k["mo_hinh_chinh"]}]'] = E.pp_mo_hinh(k['mo_hinh_chinh'])
    for id_ in them:
        pp[f'them[{id_}]'] = E.pp_mo_hinh(id_)
    return pp


def chay(tap, them):
    ds = D.SPLITS[tap]
    for a in VI_TRI:
        f = RES / f'caitien_dotre_{tap}_a{a}.json'
        cu = json.loads(f.read_text(encoding='utf-8')) if f.exists() else None
        pp = phuong_phap(a, them)
        if cu and all(t in cu for t in pp):
            print(f'[co san] {f.name}')
            continue
        if cu:                                       # chi chay phan con thieu (vd them mo hinh moi), giu ket qua cu
            pp = {t: v for t, v in pp.items() if t not in cu}
        print(f'{tap} a={a}: cho {cho(a)} s sau doan mat | phuong phap: {list(pp)}')
        bang, _ = E.chay(ds, pp, a=a)
        kq = E.tong_hop(bang, list(pp))
        if 'lap_chu_ky' in pp:
            kq['ghep_cap_so_voi_lap_chu_ky'] = E.ghep_cap(bang, list(pp), 'lap_chu_ky')
        if cu:
            for t in pp:
                cu[t] = kq[t]
            kq = cu
        kq['vi_tri'] = dict(a=a, b=a + D.GAP - 1, cho_sau_doan_mat_giay=cho(a), cho_tu_luc_mat_giay=cho(a) + D.GAP / D.FS,
                            trong_vung_huan_luyen=D.A_MIN <= a <= D.A_MAX)
        f.write_text(json.dumps(kq, ensure_ascii=False, indent=2), encoding='utf-8')
        E.in_bang(kq, [t for t in kq if isinstance(kq[t], dict) and 'rmse_macro' in kq[t]])


def ghep_cap_tu_doi_tuong(kq, ten):
    """Chenh lech ghep cap MAE so voi lap chu ky tinh tu MAE trung binh theo tung doi tuong (giong E.ghep_cap),
    tinh lai cho moi mo hinh ke ca mo hinh duoc them sau."""
    lck = kq['lap_chu_ky']['mae_theo_doi_tuong']
    out = {}
    for t in ten:
        if t == 'lap_chu_ky':
            continue
        d = {s: round(kq[t]['mae_theo_doi_tuong'][s] - lck[s], 4) for s in lck}
        out[t] = dict(theo_doi_tuong=d, trung_binh=round(sum(d.values()) / len(d), 4),
                      so_doi_tuong_tot_hon=sum(v < 0 for v in d.values()), tong_doi_tuong=len(d))
    return out


def chon_muc_cho(val):
    """Quy tac da chot truoc (caitien_dotre_quy_tac.json, caitien_bien_the_quy_tac.json), CHI dung validation:
    o moi muc cho chon mo hinh co MAE validation thap hon (mo hinh chinh hoac bien the); muc khuyen nghi la muc cho
    ngan nhat ma mo hinh do khong vuot 5% so voi mo hinh chinh o 3 s (a=192) va tot hon lap chu ky."""
    k = E.doc_khoa()
    chinh = f'mo_hinh_chinh[{k["mo_hinh_chinh"]}]'
    moc = val[192]['phuong_phap'][chinh]['mae_macro']
    nguong = round(1.05 * moc, 4)
    bang, dat = {}, []
    for a, v in val.items():
        pp = v['phuong_phap']
        ung_vien = {t: pp[t]['mae_macro'] for t in pp if t == chinh or t.startswith('them[')}
        tot = min(ung_vien, key=ung_vien.get)
        ok = ung_vien[tot] <= nguong and ung_vien[tot] < pp['lap_chu_ky']['mae_macro']
        bang[a] = dict(cho_giay=v['vi_tri']['cho_sau_doan_mat_giay'], mo_hinh_tot_hon=tot, mae_val=ung_vien[tot],
                       mae_val_lap_chu_ky=pp['lap_chu_ky']['mae_macro'], dat=ok)
        if ok:
            dat.append(a)
    a_chon = min(dat, key=lambda a: bang[a]['cho_giay'])
    # cung quy tac nhung chi xet mo hinh chinh (de biet phan A tu no dat den dau, chua co bien the)
    chi_chinh = [a for a, v in val.items() if v['phuong_phap'][chinh]['mae_macro'] <= nguong
                 and v['phuong_phap'][chinh]['mae_macro'] < v['phuong_phap']['lap_chu_ky']['mae_macro']]
    a_chinh = min(chi_chinh, key=lambda a: bang[a]['cho_giay'])
    return dict(moc_mae_val_3s=moc, nguong_5pt=nguong, theo_vi_tri=bang,
                a_khuyen_nghi=a_chon, cho_khuyen_nghi_giay=bang[a_chon]['cho_giay'], mo_hinh_khuyen_nghi=bang[a_chon]['mo_hinh_tot_hon'],
                a_khuyen_nghi_chi_mo_hinh_chinh=a_chinh, cho_khuyen_nghi_chi_mo_hinh_chinh=bang[a_chinh]['cho_giay'])


def gop():
    out = dict(ghi_chu='Cai tien ngoai yeu cau de cuong. Mo hinh va bo doc HR da khoa, khong huan luyen lai. '
                       'a=352, 384 nam ngoai vung huan luyen 64..320.', tap={})
    for tap in ('val', 'test'):
        out['tap'][tap] = {}
        for a in VI_TRI:
            f = RES / f'caitien_dotre_{tap}_a{a}.json'
            if not f.exists():
                continue
            kq = json.loads(f.read_text(encoding='utf-8'))
            ten = [t for t in kq if isinstance(kq[t], dict) and 'rmse_macro' in kq[t]]
            out['tap'][tap][a] = dict(vi_tri=kq['vi_tri'], so_cua_so=kq['so_cua_so'], tap_hop_le_chung=kq['tap_hop_le_chung'],
                                      sai_so_nen=kq['sai_so_nen_bo_doc']['mae_macro'],
                                      phuong_phap={t: {m: kq[t][m] for m in ('mae_macro', 'rmse_macro', 'mae_hr_pres', 'mae_hr_ecg',
                                                                             'ty_le_doc_duoc_hr', 'ty_le_du_phong', 'mae_theo_doi_tuong')}
                                                   for t in ten},
                                      ghep_cap=ghep_cap_tu_doi_tuong(kq, ten))
    # kiem tra cheo: test a<=320 phai trung khao sat vi tri da co
    vt = json.loads((RES / 'khaosat_vitri.json').read_text(encoding='utf-8'))['vi_tri']
    lech = []
    for a, v in out['tap'].get('test', {}).items():
        if str(a) in vt:
            for t, m in v['phuong_phap'].items():
                if t in vt[str(a)]['phuong_phap']:
                    lech.append(abs(m['mae_macro'] - vt[str(a)]['phuong_phap'][t]['mae_macro']))
    out['kiem_tra_trung_khao_sat_vi_tri'] = dict(so_so_sanh=len(lech), chenh_lon_nhat=max(lech) if lech else None)
    out['khuyen_nghi'] = chon_muc_cho(out['tap']['val'])
    (RES / 'caitien_dotre.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Kiem tra trung khao sat vi tri:', out['kiem_tra_trung_khao_sat_vi_tri'])
    print('Khuyen nghi (chi tu validation):', json.dumps(out['khuyen_nghi'], ensure_ascii=False))
    for tap, d in out['tap'].items():
        for a, v in d.items():
            print(tap, a, 'cho', v['vi_tri']['cho_sau_doan_mat_giay'], {t[:22]: round(m['mae_macro'], 4) for t, m in v['phuong_phap'].items()})
    print('Da ghi results/caitien_dotre.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tap', choices=['val', 'test'])
    ap.add_argument('--cho_phep_test', action='store_true')
    ap.add_argument('--mo_hinh_them', nargs='*', default=[])
    ap.add_argument('--gop', action='store_true')
    x = ap.parse_args()
    if x.gop:
        gop()
    else:
        if x.tap == 'test' and not x.cho_phep_test:
            raise SystemExit('Tap test dang khoa. Them --cho_phep_test.')
        chay(x.tap, x.mo_hinh_them)
