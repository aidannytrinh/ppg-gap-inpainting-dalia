# -*- coding: utf-8 -*-
"""Tong hop E2 tren du luoi lambda {0; 0,1; 0,5; 1} (muc 4.4 de cuong: "Chon lambda rieng bang validation").

Luot E2 dau tien chi thu {0; 0,5}. Bon luot con lai duoc chay bo sung sau thoi diem khoa, theo quy tac
da chot truoc trong results/e2_bo_sung_quy_tac.json: chon lambda co MAE validation macro nho nhat,
khong dung so lieu test. Chuong trinh nay chi doc ket qua huan luyen, so voi lua chon trong file khoa
va ghi results/e2_chon_du_luoi.json.

Dung: python e2_du_luoi.py
"""
import json
from datetime import datetime
from pathlib import Path

RES = Path(__file__).resolve().parent / 'results'
LUOI = ['0', '0.1', '0.5', '1']


def doc(i):
    return json.loads((RES / f'{i}.json').read_text(encoding='utf-8'))


if __name__ == '__main__':
    khoa = doc('khoa_cau_hinh')
    quy_tac = doc('e2_bo_sung_quy_tac')
    out = dict(quy_tac=quy_tac['quy_tac_chon'], thoi_diem_chot_quy_tac=quy_tac['thoi_diem_chot_quy_tac'],
               thoi_diem_khoa=khoa['thoi_diem'], nhom={})
    for nhom, khoa_key, ten in (('e2a_nolstm', 'e2a_bo_lstm', 'bo Bi-LSTM'), ('e2c_acc', 'e2c_them_acc', 'them ACC')):
        bang = {}
        for l in LUOI:
            d = doc(f'{nhom}_lam{l}')
            bang[l] = dict(id=d['id'], mae_val=round(d['mae_val_tot_nhat'], 5), epoch_tot_nhat=d['epoch_tot_nhat'],
                           epoch_chay=d['epoch_chay'], giay_moi_epoch=round(d['thoi_gian']['giay_moi_epoch'], 1),
                           tham_so=d['tham_so'])
        tot = min(bang, key=lambda l: bang[l]['mae_val'])
        out['nhom'][nhom] = dict(ten=ten, bang=bang, chon_du_luoi=bang[tot]['id'], chon_trong_file_khoa=khoa[khoa_key],
                                 trung_lua_chon_cu=bang[tot]['id'] == khoa[khoa_key])
        print(f'{ten}: ' + ', '.join(f'lam {l} -> {bang[l]["mae_val"]:.5f}' for l in LUOI)
              + f' | chon {bang[tot]["id"]} | file khoa {khoa[khoa_key]} | trung: {bang[tot]["id"] == khoa[khoa_key]}')
    (RES / 'e2_chon_du_luoi.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Da ghi results/e2_chon_du_luoi.json')

    # Neu lua chon tren du luoi khac file khoa goc: ghi file khoa bo sung (file khoa goc giu nguyen de truy vet).
    # evaluate.doc_khoa() se dung file nay cho hai ban boc tach (a) va (c) khi danh gia lai tren test.
    f_bs = RES / 'khoa_cau_hinh_bo_sung_e2.json'
    if not all(v['trung_lua_chon_cu'] for v in out['nhom'].values()) and not f_bs.exists():
        bs = dict(thoi_diem=datetime.now().isoformat(timespec='seconds'),
                  e2a_bo_lstm=out['nhom']['e2a_nolstm']['chon_du_luoi'],
                  e2c_them_acc=out['nhom']['e2c_acc']['chon_du_luoi'],
                  thay_cho=dict(e2a_bo_lstm=khoa['e2a_bo_lstm'], e2c_them_acc=khoa['e2c_them_acc']),
                  mo_hinh_chinh_khong_doi=khoa['mo_hinh_chinh'], e2b_lam0_khong_doi=khoa['e2b_lam0'],
                  ghi_chu=('Bo sung sau thoi diem khoa goc ' + khoa['thoi_diem'] + '. Chon lambda cho hai ban boc tach '
                           'tren du luoi {0; 0,1; 0,5; 1} theo quy tac chot truoc trong e2_bo_sung_quy_tac.json, '
                           'chi dung MAE validation, khong dung so lieu test. Mo hinh chinh, E4, E5 khong doi.'))
        f_bs.write_text(json.dumps(bs, ensure_ascii=False, indent=2), encoding='utf-8')
        print('Da ghi', f_bs.name, '->', bs['e2a_bo_lstm'], bs['e2c_them_acc'])
