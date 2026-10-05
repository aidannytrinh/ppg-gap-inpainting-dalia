# -*- coding: utf-8 -*-
"""Ghi danh sach doi tuong theo split co dinh (E1-E4) va theo 5 fold (E5), muc 4.5 de cuong
(phan dong goi: "danh sach doi tuong theo split va fold"). Chi doc hang so trong data.py,
khong phu thuoc ket qua huan luyen nao, chay duoc bat cu luc nao.

Dung: python danh_sach_split_fold.py
"""
import json
from pathlib import Path

import data as D

HERE = Path(__file__).resolve().parent

if __name__ == '__main__':
    fold_ds = []
    for k, f in enumerate(D.FOLDS, start=1):
        fold_ds.append(dict(fold=k, train=D.fold_train(f), val=f['val'], test=f['test']))

    kq = dict(
        split_co_dinh=dict(mo_ta='Dung cho E1, E2, E4: chia mot lan, test S13-15 khoa den khi qua E2/E4',
                            train=D.SPLITS['train'], val=D.SPLITS['val'], test=D.SPLITS['test']),
        fold_e5=dict(mo_ta='Kiem dinh cheo theo doi tuong muc 2.4: moi fold test 3, val 2, train 10 nguoi',
                     folds=fold_ds),
    )
    (HERE / 'results' / 'danh_sach_split_fold.json').write_text(
        json.dumps(kq, ensure_ascii=False, indent=2), encoding='utf-8')

    # kiem tra khong trung: trong moi fold, va giua split co dinh, khong nguoi nao vua train vua test/val
    for f in fold_ds:
        assert not (set(f['train']) & set(f['val']) & set(f['test']))
        assert sorted(f['train'] + f['val'] + f['test']) == list(range(1, 16))
    assert sorted(D.SPLITS['train'] + D.SPLITS['val'] + D.SPLITS['test']) == list(range(1, 16))

    print('Da ghi results/danh_sach_split_fold.json')
    print('Split co dinh: train', D.SPLITS['train'], '| val', D.SPLITS['val'], '| test', D.SPLITS['test'])
    for f in fold_ds:
        print(f'Fold {f["fold"]}: train {f["train"]} | val {f["val"]} | test {f["test"]}')
