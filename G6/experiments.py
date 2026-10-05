# -*- coding: utf-8 -*-
"""Dieu phoi thi nghiem E1 den E5 theo muc 4.4 va thu tu bat buoc o muc 5.1 de cuong.

  E4  khao sat lambda trong {0; 0,1; 0,5; 1} tren validation, chia co dinh. Chot lambda. Khong nhin test.
  E2  boc tach, huan luyen rieng cung ngan sach: (a) bo Bi-LSTM, (b) lambda=0 (dung lai tu E4), (c) them ACC.
      (a) va (c) chon lambda rieng bang validation.
  KHOA cau hinh: ghi results/khoa_cau_hinh.json kem thoi diem. Sau day khong doi sieu tham so.
  E1  so mo hinh voi ba doi chuan tren test S13-S15. Chi chay duoc khi da khoa.
  E3  phan tich theo tung hoat dong tren ket qua E1, quy tac 80%.
  E5  kiem dinh cheo 5 fold, lambda chon trong validation rieng cua tung fold.

Moi luot huan luyen ghi results/<id>.json va bo qua neu da co, nen ngat giua chung chay lai khong mat cong.

Thu hep luoi lambda o E2 va E5 (muc 5.1: "khi khong du tai nguyen, ghi ro gioi han va xin dieu chinh
pham vi"): T4 mien phi chi 49 giay/epoch, du 4 x 2 (E2) + 4 x 5 (E5) = 28 luot se mat gan 30 gio, khong
kip 2 tuan. Neu KHONG truyen --luoi, E2 va E5 tu doc ket qua E4, dung {0, lambda tot nhat} cho E2 va chi
{lambda tot nhat} cho E5 (5 luot thay vi 20). Muon do du luoi nhu cu thi tu truyen --luoi 0 0.1 0.5 1.

Dung:
  python experiments.py e4                 4 luot lambda tren chia co dinh (bat buoc du, khong thu hep)
  python experiments.py e2 --luoi 0 0.1 0.5 1   boc tach (a) va (c) tren du luoi (cach da dung trong bao cao,
                                                xem e2_du_luoi.py); khong truyen --luoi thi tu thu hep {0, lambda E4}
  python experiments.py khoa               chot lambda va cau hinh tu ket qua E4/E2, ghi file khoa
  python experiments.py e1                 danh gia tren test (can file khoa)
  python experiments.py e5                 5 fold, tu dung lambda tot nhat cua E4 (5 luot)
  python experiments.py e5 --luoi 0 0.1 0.5 1   do du luoi cho ca 5 fold (20 luot, can nhieu gio)
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
KHOA = RES / 'khoa_cau_hinh.json'
LUOI_LAM = [0.0, 0.1, 0.5, 1.0]
PY = sys.executable


def ten_lam(l):
    return str(l).rstrip('0').rstrip('.') if '.' in str(l) else str(l)


def chay(id_, lam, **k):
    """Goi train.py nhu mot tien trinh rieng, de moi luot doc lap va ghi ket qua rieng."""
    if (RES / f'{id_}.json').exists():
        print(f'  [co san] {id_}')
        return json.loads((RES / f'{id_}.json').read_text(encoding='utf-8'))
    cmd = [PY, '-u', str(HERE / 'train.py'), '--id', id_, '--lam', str(lam)]
    if k.get('khong_lstm'): cmd.append('--khong_lstm')
    if k.get('acc'): cmd.append('--acc')
    if k.get('fold'): cmd += ['--fold', str(k['fold'])]
    print('  >', ' '.join(cmd[2:]))
    subprocess.run(cmd, check=True, cwd=HERE)
    return json.loads((RES / f'{id_}.json').read_text(encoding='utf-8'))


def tot_nhat(ids):
    """Chon luot co MAE validation macro thap nhat trong danh sach id."""
    kq = {i: json.loads((RES / f'{i}.json').read_text(encoding='utf-8')) for i in ids if (RES / f'{i}.json').exists()}
    if not kq:
        return None, {}
    best = min(kq, key=lambda i: kq[i]['mae_val_tot_nhat'])
    return best, {i: r['mae_val_tot_nhat'] for i, r in kq.items()}


def lam_tot_nhat_e4():
    """Doc lambda tot nhat tu ket qua E4 da chay. Bao loi ro rang neu E4 chua xong."""
    best, bang = tot_nhat([f'e4_lam{ten_lam(l)}' for l in LUOI_LAM])
    if best is None or len(bang) < len(LUOI_LAM):
        raise SystemExit(f'Can chay xong ca 4 luot E4 truoc (hien co {len(bang)}/4). Chay: python experiments.py e4')
    return json.loads((RES / f'{best}.json').read_text(encoding='utf-8'))['cau_hinh']['lam']


# ------------------------------------------------------------------ E4
def e4(luoi=None):
    luoi = luoi or LUOI_LAM
    print('E4: khao sat lambda tren validation, chia co dinh. Luoi:', luoi)
    for l in luoi:
        chay(f'e4_lam{ten_lam(l)}', l)
    best, bang = tot_nhat([f'e4_lam{ten_lam(l)}' for l in luoi])
    print('  MAE_val theo lambda:', {k: round(v, 5) for k, v in bang.items()}, '| tot nhat:', best)


# ------------------------------------------------------------------ E2
def e2(luoi=None):
    if luoi is None:
        l_e4 = lam_tot_nhat_e4()
        luoi = sorted({0.0, l_e4})
        print(f'E2: khong truyen --luoi, tu dung luoi thu hep {luoi} (0 va lambda tot nhat cua E4 = {l_e4})')
    else:
        print('E2: boc tach thanh phan, luoi duoc chi dinh', luoi)
    for l in luoi:
        chay(f'e2a_nolstm_lam{ten_lam(l)}', l, khong_lstm=True)
    for l in luoi:
        chay(f'e2c_acc_lam{ten_lam(l)}', l, acc=True)
    for nhom in ('e2a_nolstm', 'e2c_acc'):
        best, bang = tot_nhat([f'{nhom}_lam{ten_lam(l)}' for l in luoi])
        print(f'  {nhom}: MAE_val theo lambda', {k: round(v, 5) for k, v in bang.items()}, '| tot nhat:', best)
    print('  (b) lambda=0 dung lai e4_lam0')


# ------------------------------------------------------------------ KHOA
def khoa():
    if KHOA.exists():
        print('Da khoa tu', json.loads(KHOA.read_text(encoding='utf-8'))['thoi_diem'], '. Khong ghi de.')
        return
    chinh, bang_chinh = tot_nhat([f'e4_lam{ten_lam(l)}' for l in LUOI_LAM])
    if chinh is None:
        raise SystemExit('Chua co ket qua E4, khong the khoa.')
    a_, bang_a = tot_nhat([f'e2a_nolstm_lam{ten_lam(l)}' for l in LUOI_LAM])
    c_, bang_c = tot_nhat([f'e2c_acc_lam{ten_lam(l)}' for l in LUOI_LAM])
    kq = dict(thoi_diem=datetime.now().isoformat(timespec='seconds'),
              mo_hinh_chinh=chinh, mae_val_e4=bang_chinh,
              e2a_bo_lstm=a_, mae_val_e2a=bang_a,
              e2b_lam0='e4_lam0',
              e2c_them_acc=c_, mae_val_e2c=bang_c,
              ghi_chu='Chot bang MAE validation macro. Sau thoi diem nay khong doi sieu tham so. Test chua duoc cham.')
    KHOA.write_text(json.dumps(kq, ensure_ascii=False, indent=2), encoding='utf-8')
    print('DA KHOA cau hinh luc', kq['thoi_diem'])
    print('  mo hinh chinh:', chinh, '| E2a:', a_, '| E2c:', c_)


# ------------------------------------------------------------------ E1, E3
def e1():
    if not KHOA.exists():
        raise SystemExit('Chua khoa cau hinh. Chay: python experiments.py khoa')
    cmd = [PY, '-u', str(HERE / 'evaluate.py'), '--mohinh', '--tap', 'test', '--cho_phep_test']
    print('E1 + E3: danh gia tren test S13-S15 theo file khoa')
    subprocess.run(cmd, check=True, cwd=HERE)


# ------------------------------------------------------------------ E5
def e5(luoi=None):
    if luoi is None:
        l_e4 = lam_tot_nhat_e4()
        luoi = [l_e4]
        print(f'E5: khong truyen --luoi, tu dung 1 gia tri lambda = {l_e4} (tot nhat cua E4) cho ca 5 fold, '
              f'theo muc 5.1: thu hep khi khong du tai nguyen, khong do lai lua chon kien truc bang outer test.')
    else:
        print('E5: kiem dinh cheo 5 fold, luoi lambda duoc chi dinh', luoi)
    tong = {}
    for f in range(1, 6):
        for l in luoi:
            chay(f'e5_f{f}_lam{ten_lam(l)}', l, fold=f)
        best, bang = tot_nhat([f'e5_f{f}_lam{ten_lam(l)}' for l in luoi])
        tong[f] = dict(tot_nhat=best, mae_val=bang)
        print(f'  fold {f}: tot nhat {best}', {k: round(v, 5) for k, v in bang.items()})
    (RES / 'e5_chon_theo_fold.json').write_text(json.dumps(dict(luoi=luoi, fold=tong), ensure_ascii=False, indent=2), encoding='utf-8')
    print('  Da ghi e5_chon_theo_fold.json. Danh gia ngoai mau: python evaluate.py --e5 (buoc sau)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('buoc', choices=['e4', 'e2', 'khoa', 'e1', 'e5'])
    ap.add_argument('--luoi', type=float, nargs='+', default=None,
                    help='luoi lambda. E4: mac dinh 0 0.1 0.5 1. E2/E5: mac dinh None de tu thu hep theo E4.')
    a = ap.parse_args()
    t0 = time.time()
    {'e4': lambda: e4(a.luoi), 'e2': lambda: e2(a.luoi), 'khoa': khoa, 'e1': e1, 'e5': lambda: e5(a.luoi)}[a.buoc]()
    print(f'Tong thoi gian buoc {a.buoc}: {(time.time()-t0)/60:.1f} phut')
