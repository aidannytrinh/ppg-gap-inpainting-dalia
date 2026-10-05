# -*- coding: utf-8 -*-
"""Chi so danh gia, muc 4.1, 4.2 va E3 muc 4.4 de cuong.

  - CT (10): MAE_i va RMSE_i tinh RIENG tren 128 mau bi che cua tung cua so.
  - CT (11): gop macro: trung binh trong tung doi tuong roi trung binh deu giua cac doi tuong.
  - CT (13): MAE_HR,pres = trung binh |H(r_out) - H(r)|.
  - CT (14): MAE_HR,ECG  = trung binh |H(r_out) - h_ECG|.
  - Muc 4.2: luon kem H(PPG day du) so voi ECG (sai so nen). Doi chieu nhieu phuong phap tren
             TAP CUA SO HOP LE CHUNG, cong bo ty le doc duoc HR cua tung phuong phap.
  - Muc 4.1: chenh lech ghep cap tren cung doi tuong.
  - E3     : bao cao theo tung hoat dong; cua so khong du 80% mot nhan la hon hop/chuyen tiep,
             KHONG bi loai khoi chi so chung.

An toan giao thuc: tap test bi khoa, phai truyen --cho_phep_test (muc 5.1).

Dung:
  python evaluate.py --doichuan --tap val                 ba doi chuan tren validation
  python evaluate.py --mohinh --tap test --cho_phep_test  E1 + E3: doi chuan + cac mo hinh trong file khoa
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

import baselines as B
import data as D
import hr_reader as H

RES = Path(__file__).resolve().parent / 'results'
W = Path(__file__).resolve().parent / 'weights'
KHOA = RES / 'khoa_cau_hinh.json'
KHOA_BO_SUNG = RES / 'khoa_cau_hinh_bo_sung_e2.json'


def doc_khoa():
    """Cau hinh da khoa. File khoa goc giu nguyen; neu co lua chon E2 bo sung (chon lambda tren du luoi,
    chi bang validation, xem e2_du_luoi.py) thi thay hai ban boc tach (a) va (c) theo file bo sung."""
    k = json.loads(KHOA.read_text(encoding='utf-8'))
    if KHOA_BO_SUNG.exists():
        bs = json.loads(KHOA_BO_SUNG.read_text(encoding='utf-8'))
        k = dict(k, e2a_bo_lstm=bs['e2a_bo_lstm'], e2c_them_acc=bs['e2c_them_acc'])
    return k


# ------------------------------------------------------------------ chi so
def chi_so_gap(Z, Y, M):
    """Cong thuc (10). Z, Y dang (N, 512)."""
    g = M == 1
    e = Z[:, g] - Y[:, g]
    return np.abs(e).mean(axis=1), np.sqrt((e ** 2).mean(axis=1))


def gop_macro(gt, sid):
    """Cong thuc (11)."""
    gt, sid = np.asarray(gt, dtype=np.float64), np.asarray(sid)
    theo = {int(s): float(np.nanmean(gt[sid == s])) for s in np.unique(sid) if np.isfinite(gt[sid == s]).any()}
    return (float(np.mean(list(theo.values()))) if theo else float('nan')), theo


def chuan_hoa_lo(R, M):
    """CT (3)(4) tren ca lo voi cung mot mat na."""
    O = M == 0
    Q = R[:, O]
    mu = Q.mean(axis=1)
    sg = np.sqrt(((Q - mu[:, None]) ** 2).mean(axis=1))
    s = np.maximum(sg, D.EPS_SIGMA)
    Z = (R - mu[:, None]) / s[:, None]
    return Z, Z * (1.0 - M), mu, s


# ---------------------------------------------------------- cac phuong phap
def pp_doi_chuan(ten):
    f = B.DOI_CHUAN[ten]

    def du_doan(R, Z_in, M, A, ch):
        Y, dp = np.empty_like(Z_in), np.zeros(len(Z_in), dtype=bool)
        for j in range(len(Z_in)):
            Y[j], dp[j] = f(Z_in[j], M, ch)
        return Y, dp
    return du_doan


def pp_mo_hinh(id_):
    """Du doan bang mo hinh da huan luyen, dung dung duong ong tien xu ly cua train.py."""
    import torch
    import train as T
    from model import InpaintAE
    cfg = json.loads((RES / f'{id_}.json').read_text(encoding='utf-8'))['cau_hinh']
    m = InpaintAE(5 if cfg['acc'] else 2, dung_lstm=cfg['dung_lstm'])
    m.load_state_dict(torch.load(W / f'{id_}.pt', map_location='cpu'))
    m.eval()

    @torch.no_grad()
    def du_doan(R, Z_in, M, A, ch):
        a = int(np.flatnonzero(M == 1)[0])
        Rt = torch.from_numpy(R).float()
        At = torch.from_numpy(A).float() if cfg['acc'] else None
        Y = np.empty_like(Z_in)
        for lo in range(0, len(Rt), 512):
            u, z, Mt = T.lam_dau_vao(Rt[lo:lo + 512], torch.full((len(Rt[lo:lo + 512]),), a),
                                     At[lo:lo + 512] if At is not None else None)
            Y[lo:lo + 512] = m(u).numpy()
        return Y, np.zeros(len(Y), dtype=bool)
    return du_doan


# ------------------------------------------------------------------ chay
def chay(ds, phuong_phap, a=D.A_EVAL, can_acc=False, ch=None):
    """Chay moi phuong phap tren danh sach doi tuong. Tra ve bang theo cua so va du doan vung thieu.

    ch: cau hinh bo doc HR. Mac dinh la bo doc chon tren S1-S12; E5 truyen bo doc rieng cua fold.
    """
    ch = ch or H.tai_cau_hinh()
    M = D.mat_na(a)
    g = M == 1
    cot = {k: [] for k in ('sid', 'win', 'giay_bat_dau', 'act', 'hr_ecg', 'hr_day_du')}
    for k in phuong_phap:
        for c in ('mae', 'rmse', 'hr', 'duphong'):
            cot[f'{c}_{k}'] = []
    du_doan_gap = {k: [] for k in phuong_phap}
    dich_gap = []

    for sid in ds:
        d = D.chi_muc(sid, can_acc=can_acc)
        idx = np.flatnonzero(d['du_dk'])
        R = np.stack([D.cua_so(d['bvp'], i) for i in idx]).astype(np.float64)
        A = np.stack([D.acc_cua_so(d['acc'], i) for i in idx]) if can_acc else None
        Z, Z_in, mu, s = chuan_hoa_lo(R, M)
        cot['sid'] += [sid] * len(idx)
        cot['win'] += idx.tolist()
        cot['giay_bat_dau'] += (idx * D.STEP / D.FS).tolist()
        cot['act'] += d['act_win'][idx].tolist()
        cot['hr_ecg'] += d['hr'][idx].tolist()
        cot['hr_day_du'] += H.doc_hr(R, ch).tolist()
        dich_gap.append(Z[:, g].astype(np.float32))

        for ten, f in phuong_phap.items():
            Y, dp = f(R, Z_in, M, A, ch)
            mae, rmse = chi_so_gap(Z, Y, M)
            R_out = (1.0 - M) * R + M * (mu[:, None] + s[:, None] * Y)   # CT (6)
            cot[f'mae_{ten}'] += mae.tolist()
            cot[f'rmse_{ten}'] += rmse.tolist()
            cot[f'hr_{ten}'] += H.doc_hr(R_out, ch).tolist()
            cot[f'duphong_{ten}'] += dp.astype(int).tolist()
            du_doan_gap[ten].append(Y[:, g].astype(np.float32))
        print(f'  S{sid}: {len(idx)} cua so, {len(phuong_phap)} phuong phap')

    bang = {k: np.asarray(v) for k, v in cot.items()}
    gap = dict(dich=np.concatenate(dich_gap), **{k: np.concatenate(v) for k, v in du_doan_gap.items()})
    return bang, gap


def tong_hop(bang, ten_pp, mask=None):
    """Gop theo CT (11) va hai chi so HR tren tap hop le chung. mask: chon tap con cua so (dung cho E3)."""
    sel = np.ones(len(bang['sid']), dtype=bool) if mask is None else mask
    sid = bang['sid'][sel]
    chung = np.isfinite(bang['hr_ecg'][sel]) & np.isfinite(bang['hr_day_du'][sel])
    for t in ten_pp:
        chung &= np.isfinite(bang[f'hr_{t}'][sel])
    kq = dict(so_cua_so=int(sel.sum()), so_doi_tuong=int(len(np.unique(sid))),
              tap_hop_le_chung=int(chung.sum()),
              ty_le_hop_le_chung=round(float(chung.mean()), 4) if sel.any() else None)
    nen, nen_ng = gop_macro(np.abs(bang['hr_day_du'][sel] - bang['hr_ecg'][sel])[chung], sid[chung])
    kq['sai_so_nen_bo_doc'] = dict(mae_macro=round(nen, 3), theo_doi_tuong={k: round(v, 3) for k, v in nen_ng.items()})
    for t in ten_pp:
        mae_m, mae_ng = gop_macro(bang[f'mae_{t}'][sel], sid)
        rms_m, rms_ng = gop_macro(bang[f'rmse_{t}'][sel], sid)
        pres, pres_ng = gop_macro(np.abs(bang[f'hr_{t}'][sel] - bang['hr_day_du'][sel])[chung], sid[chung])
        ecg, ecg_ng = gop_macro(np.abs(bang[f'hr_{t}'][sel] - bang['hr_ecg'][sel])[chung], sid[chung])
        kq[t] = dict(mae_macro=round(mae_m, 4), rmse_macro=round(rms_m, 4),
                     mae_theo_doi_tuong={k: round(v, 4) for k, v in mae_ng.items()},
                     rmse_theo_doi_tuong={k: round(v, 4) for k, v in rms_ng.items()},
                     mae_hr_pres=round(pres, 3), mae_hr_ecg=round(ecg, 3),
                     mae_hr_pres_theo_doi_tuong={k: round(v, 3) for k, v in pres_ng.items()},
                     mae_hr_ecg_theo_doi_tuong={k: round(v, 3) for k, v in ecg_ng.items()},
                     ty_le_doc_duoc_hr=round(float(np.isfinite(bang[f'hr_{t}'][sel]).mean()), 4),
                     ty_le_du_phong=round(float(bang[f'duphong_{t}'][sel].mean()), 4))
    return kq


def ghep_cap(bang, ten_pp, chuan):
    """Muc 4.1: chenh lech ghep cap MAE theo doi tuong so voi phuong phap 'chuan' (am la tot hon)."""
    out = {}
    for t in ten_pp:
        if t == chuan:
            continue
        _, a = gop_macro(bang[f'mae_{t}'], bang['sid'])
        _, b = gop_macro(bang[f'mae_{chuan}'], bang['sid'])
        d = {s: round(a[s] - b[s], 4) for s in a}
        out[t] = dict(so_voi=chuan, theo_doi_tuong=d, trung_binh=round(float(np.mean(list(d.values()))), 4),
                      so_doi_tuong_tot_hon=int(sum(v < 0 for v in d.values())), tong_doi_tuong=len(d))
    return out


def e3_theo_hoat_dong(bang, ten_pp):
    """E3: tach theo tung hoat dong va nhom hon hop/chuyen tiep. Khong loai cua so nao."""
    out = {}
    for ma in sorted(set(bang['act'].tolist())):
        ten = 'hon hop/chuyen tiep' if ma == D.HON_HOP else D.HOAT_DONG.get(int(ma), str(ma))
        kq = tong_hop(bang, ten_pp, mask=bang['act'] == ma)
        out[ten] = {k: kq[k] for k in ('so_cua_so', 'so_doi_tuong', 'tap_hop_le_chung', 'sai_so_nen_bo_doc')}
        for t in ten_pp:
            out[ten][t] = {k: kq[t][k] for k in ('mae_macro', 'rmse_macro', 'mae_hr_pres', 'mae_hr_ecg')}
    return out


def ghi(bang, gap, kq, ten):
    RES.mkdir(exist_ok=True)
    khoa = list(bang)
    with open(RES / f'{ten}_theo_cuaso.csv', 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(khoa)
        w.writerows(zip(*[bang[k] for k in khoa]))
    np.savez_compressed(RES / f'{ten}_dudoan_gap.npz', sid=bang['sid'], win=bang['win'], **gap)
    (RES / f'{ten}.json').write_text(json.dumps(kq, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Da ghi:', f'{ten}.json, {ten}_theo_cuaso.csv, {ten}_dudoan_gap.npz')


def in_bang(kq, ten_pp):
    print('\n%-22s %8s %8s %10s %10s %8s' % ('phuong phap', 'MAE', 'RMSE', 'MAE_HR_pr', 'MAE_HR_ECG', 'du_phong'))
    for t in ten_pp:
        r = kq[t]
        print('%-22s %8.4f %8.4f %10.3f %10.3f %7.1f%%' % (
            t, r['mae_macro'], r['rmse_macro'], r['mae_hr_pres'], r['mae_hr_ecg'], r['ty_le_du_phong'] * 100))
    print('Sai so nen bo doc H(PPG day du) so ECG: %.3f bpm | tap hop le chung %d/%d'
          % (kq['sai_so_nen_bo_doc']['mae_macro'], kq['tap_hop_le_chung'], kq['so_cua_so']))


# ------------------------------------------------------------------ E5
def danh_gia_e5():
    """Danh gia ngoai mau cho kiem dinh cheo 5 fold, muc 2.4.

    Moi fold: bo doc HR chon lai tren train + validation cua chinh fold (khong dung test cua fold),
    mo hinh la luot co MAE validation tot nhat trong fold, danh gia tren 3 doi tuong test cua fold
    cung ba doi chuan. Bao cao: MAE macro tung fold, trung binh va do lech chuan qua 5 fold,
    ket qua ngoai mau cua du 15 doi tuong, chenh lech ghep cap voi lap chu ky.
    """
    chon = json.loads((RES / 'e5_chon_theo_fold.json').read_text(encoding='utf-8'))
    tong = dict(luoi=chon['luoi'], fold={}, theo_doi_tuong={})
    for k, f in enumerate(D.FOLDS, 1):
        id_ = chon['fold'][str(k)]['tot_nhat']
        f_bd = RES / f'bodoc_hr_f{k}.json'
        if not f_bd.exists():
            print(f'\nFold {k}: chon tham so bo doc HR tren train + val cua fold')
            H.chon_tham_so(ds=D.fold_train(f) + f['val'], f_out=f_bd)
        ch = json.loads(f_bd.read_text(encoding='utf-8'))
        print(f'\nFold {k}: test S{f["test"]} | mo hinh {id_} | bo doc HR_max={ch["hr_max"]} prom={ch["prominence_ty_le"]}')
        pp = {t: pp_doi_chuan(t) for t in B.DOI_CHUAN}
        pp['mo_hinh'] = pp_mo_hinh(id_)
        bang, gap = chay(f['test'], pp, ch=ch)
        kq = tong_hop(bang, list(pp))
        kq['ghep_cap_so_voi_lap_chu_ky'] = ghep_cap(bang, list(pp), 'lap_chu_ky')
        kq['mo_hinh_id'] = id_
        kq['bo_doc_hr'] = {k2: ch[k2] for k2 in ('hr_max', 'distance_mau', 'prominence_ty_le', 'mae_nen')}
        ghi(bang, gap, kq, f'e5_f{k}_test')
        tong['fold'][k] = {t: {m: kq[t][m] for m in ('mae_macro', 'rmse_macro', 'mae_hr_pres', 'mae_hr_ecg')}
                           for t in pp}
        tong['fold'][k]['mo_hinh_id'] = id_
        for s in f['test']:
            tong['theo_doi_tuong'][s] = {t: kq[t]['mae_theo_doi_tuong'][s] for t in pp}

    ten_pp = list(tong['fold'][1].keys() - {'mo_hinh_id'})
    tong['trung_binh_5_fold'] = {}
    for t in sorted(ten_pp):
        for m in ('mae_macro', 'rmse_macro', 'mae_hr_pres', 'mae_hr_ecg'):
            v = np.array([tong['fold'][k][t][m] for k in tong['fold']])
            tong['trung_binh_5_fold'].setdefault(t, {})[m] = dict(tb=round(float(v.mean()), 4),
                                                                  dlc=round(float(v.std(ddof=1)), 4))
    d = np.array([tong['theo_doi_tuong'][s]['mo_hinh'] - tong['theo_doi_tuong'][s]['lap_chu_ky']
                  for s in sorted(tong['theo_doi_tuong'])])
    tong['ghep_cap_15_doi_tuong'] = dict(so_voi='lap_chu_ky', trung_binh=round(float(d.mean()), 4),
                                         so_doi_tuong_tot_hon=int((d < 0).sum()), tong=len(d))
    (RES / 'e5_tonghop.json').write_text(json.dumps(tong, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n=== E5: trung binh +- do lech chuan qua 5 fold ===')
    for t in sorted(ten_pp):
        r = tong['trung_binh_5_fold'][t]
        print('%-12s MAE %.4f +- %.4f | RMSE %.4f +- %.4f | MAE_HR_pres %.3f +- %.3f'
              % (t, r['mae_macro']['tb'], r['mae_macro']['dlc'], r['rmse_macro']['tb'], r['rmse_macro']['dlc'],
                 r['mae_hr_pres']['tb'], r['mae_hr_pres']['dlc']))
    g = tong['ghep_cap_15_doi_tuong']
    print('Mo hinh tot hon lap chu ky o %d/%d doi tuong, chenh lech MAE trung binh %.4f'
          % (g['so_doi_tuong_tot_hon'], g['tong'], g['trung_binh']))
    print('Da ghi: e5_tonghop.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--e5', action='store_true', help='danh gia ngoai mau 5 fold (can e5_chon_theo_fold.json)')
    ap.add_argument('--doichuan', action='store_true')
    ap.add_argument('--mohinh', action='store_true', help='doi chuan + cac mo hinh trong file khoa (E1, E3)')
    ap.add_argument('--tap', default='val', choices=['train', 'val', 'test'])
    ap.add_argument('--cho_phep_test', action='store_true')
    a = ap.parse_args()
    if a.e5:
        danh_gia_e5()
        raise SystemExit(0)
    if a.tap == 'test' and not a.cho_phep_test:
        raise SystemExit('Tap test dang khoa. Muc 5.1: chi chay test sau khi da khoa cau hinh. Them --cho_phep_test.')
    if a.mohinh and not KHOA.exists():
        raise SystemExit('Chua co results/khoa_cau_hinh.json. Chay: python experiments.py khoa')

    pp = {t: pp_doi_chuan(t) for t in B.DOI_CHUAN}
    can_acc = False
    if a.mohinh:
        k = doc_khoa()
        for nhan, id_ in (('mo_hinh_chinh', k['mo_hinh_chinh']), ('e2a_bo_lstm', k['e2a_bo_lstm']),
                          ('e2b_lam0', k['e2b_lam0']), ('e2c_them_acc', k['e2c_them_acc'])):
            if id_:
                pp[f'{nhan}[{id_}]'] = pp_mo_hinh(id_)
                can_acc |= json.loads((RES / f'{id_}.json').read_text(encoding='utf-8'))['cau_hinh']['acc']
    ds = D.SPLITS[a.tap]
    ten = ('e1' if a.mohinh else 'doichuan') + f'_{a.tap}'
    print(f'Danh gia tren tap {a.tap}: S{", S".join(map(str, ds))} | phuong phap: {list(pp)}')
    bang, gap = chay(ds, pp, can_acc=can_acc)
    kq = tong_hop(bang, list(pp))
    kq['ghep_cap_so_voi_lap_chu_ky'] = ghep_cap(bang, list(pp), 'lap_chu_ky')
    if a.mohinh:
        kq['e3_theo_hoat_dong'] = e3_theo_hoat_dong(bang, list(pp))
    ghi(bang, gap, kq, ten)
    in_bang(kq, list(pp))
