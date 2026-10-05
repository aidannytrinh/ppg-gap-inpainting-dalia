# -*- coding: utf-8 -*-
"""Nap du lieu, cat cua so, sinh mat na, chuan hoa theo phan quan sat, chia tap va fold.

Bam theo de cuong da chinh sua:
  - Cong thuc (2): so cua so N_s = max(0, floor((n_s - 512)/128) + 1), khong noi doan roi.
  - Muc 2.3   : train lay a ngau nhien deu trong 64..320; val/test dung khoang thieu co dinh 192..319.
  - Cong thuc (3)(4): mu, sigma tinh CHI tren phan quan sat O = {t: M_t = 0}; s_O = max(sigma, 1e-6);
                      mau thieu dat bang 0 SAU chuan hoa; dich tai tao dung cung mu va s_O.
  - Cong thuc (1): dau vao u = [z_in, M] voi z_in = (1 - M) * z.
  - Muc 2.2   : cua so khong huu han hoac sigma_O <= 1e-6 la khong du dieu kien, phai cong bo so luong.
  - Muc 2.4   : bang 5 fold theo doi tuong.
  - Muc 4.4 E3: cua so duoc gan hoat dong khi mot nhan chiem it nhat 80% thoi luong, con lai la hon hop.

Dung: python data.py --thongke   -> xuat bang thong ke theo doi tuong ra results/thongke_dulieu.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MOD = HERE.parent
CACHE = MOD / 'dalia_cache'
RES = HERE / 'results'

FS = 64                      # Hz, tan so BVP
FS_ACC = 32                  # Hz, tan so ACC co tay
FS_ACT = 4                   # Hz, tan so nhan hoat dong
WIN, STEP = 512, 128         # 8 giay, buoc 2 giay
GAP = 128                    # 2 giay bi che
A_MIN, A_MAX = 64, 320       # vi tri bat dau mat na khi huan luyen (bao gom ca hai dau)
A_EVAL = 192                 # khoang thieu co dinh 192..319 khi validation va test
EPS_SIGMA = 1e-6
NGUONG_HD = 0.8              # quy tac 80% cua muc 4.4

SPLITS = {
    'train': list(range(1, 11)),    # S1..S10
    'val':   [11, 12],              # S11, S12
    'test':  [13, 14, 15],          # S13..S15
}

# Bang fold o muc 2.4. Moi fold chia du 15 doi tuong thanh test 3, validation 2, train 10.
FOLDS = [
    {'test': [1, 2, 3],    'val': [4, 5]},
    {'test': [4, 5, 6],    'val': [7, 8]},
    {'test': [7, 8, 9],    'val': [10, 11]},
    {'test': [10, 11, 12], 'val': [13, 14]},
    {'test': [13, 14, 15], 'val': [1, 2]},
]

# Ten hoat dong theo readme chinh thuc cua PPG-DaLiA (ID 0 la chuyen tiep).
HOAT_DONG = {0: 'chuyen tiep', 1: 'ngoi', 2: 'cau thang', 3: 'bi lac', 4: 'dap xe',
             5: 'lai xe', 6: 'an trua', 7: 'di bo', 8: 'lam viec'}
HON_HOP = -1                 # ma rieng cho cua so hon hop hoac chuyen tiep


def fold_train(f):
    """Danh sach doi tuong huan luyen cua mot fold: 15 nguoi tru test va validation."""
    bo = set(f['test']) | set(f['val'])
    return [s for s in range(1, 16) if s not in bo]


def so_cua_so(n):
    """Cong thuc (2) trong de cuong."""
    return max(0, (n - WIN) // STEP + 1)


# ----------------------------------------------------------------------- nap
def nap_doi_tuong(sid, can_acc=False):
    """Nap mot doi tuong tu cache. Tra ve BVP goc (chua loc), nhan hoat dong, nhan HR tu ECG."""
    d = np.load(CACHE / f'dalia_S{sid}.npz')
    h = np.load(CACHE / f'hr_S{sid}.npz')
    out = {'sid': sid,
           'bvp': np.asarray(d['bvp'], dtype=np.float32),
           'act': np.asarray(d['act'], dtype=np.int16),
           'hr': np.asarray(h['hr'], dtype=np.float32)}
    n_win = so_cua_so(len(out['bvp']))
    if len(out['hr']) != n_win:
        raise ValueError(f'S{sid}: nhan HR co {len(out["hr"])} phan tu, cua so tinh duoc {n_win}')
    if can_acc:
        out['acc'] = np.asarray(np.load(CACHE / f'imu_S{sid}.npz')['acc_w'], dtype=np.float32)
    return out


def cua_so(bvp, i):
    """Lat cat cua so thu i: mau [i*128, i*128+512). Khong sao chep du lieu."""
    return bvp[i * STEP: i * STEP + WIN]


def acc_cua_so(acc, i):
    """ACC 32 Hz cua cua so thu i, noi suy tuyen tinh theo thoi gian len 64 Hz (muc 2.4)."""
    lo = i * STEP // 2                       # 128 mau o 64 Hz = 64 mau o 32 Hz
    seg = acc[lo: lo + WIN // 2]             # (256, 3)
    if len(seg) < WIN // 2:
        return None
    t_cu = np.arange(WIN // 2, dtype=np.float64) / FS_ACC
    t_moi = np.arange(WIN, dtype=np.float64) / FS
    return np.stack([np.interp(t_moi, t_cu, seg[:, c]) for c in range(3)], axis=1).astype(np.float32)


# ------------------------------------------------------------------- mat na
def mat_na(a):
    """Mat na M: 1 tai doan thieu a..a+127, 0 tai cac mau quan sat duoc."""
    M = np.zeros(WIN, dtype=np.float32)
    M[a: a + GAP] = 1.0
    return M


def a_ngau_nhien(rng):
    """Vi tri bat dau khi huan luyen: nguyen, deu trong 64..320 ke ca hai dau (muc 2.3)."""
    return int(rng.integers(A_MIN, A_MAX + 1))


# ----------------------------------------------------------------- chuan hoa
def chuan_hoa(r, M):
    """Cong thuc (3) va (4). Tra ve z (dich tai tao, da chuan hoa tren ca 512 mau),
    z_in (dau vao, mau thieu dat bang 0), mu, s.

    Thong ke chi tinh tren O = {t: M_t = 0}. Dich tai tao dung cung mu va s.
    """
    O = M == 0
    quan_sat = r[O]
    mu = float(quan_sat.mean())
    sigma = float(np.sqrt(((quan_sat - mu) ** 2).mean()))
    s = max(sigma, EPS_SIGMA)
    z = ((r - mu) / s).astype(np.float32)
    z_in = z * (1.0 - M)                     # cong thuc (1): mau thieu bang 0 SAU chuan hoa
    return z, z_in, mu, s, sigma


def dau_vao(z_in, M):
    """Cong thuc (1): u = [z_in, M], kich thuoc (512, 2)."""
    return np.stack([z_in, M], axis=1)


def ghep_z(z_hat, z, M):
    """Cong thuc (5): z_out = (1 - M) * z + M * z_hat. Phan quan sat khong bi thay the."""
    return (1.0 - M) * z + M * z_hat


def ve_don_vi_goc(z_hat, r, M, mu, s):
    """Cong thuc (6): r_out = (1 - M) * r + M * (mu + s * z_hat)."""
    return (1.0 - M) * r + M * (mu + s * z_hat)


# ------------------------------------------------------------- nhan hoat dong
def nhan_hoat_dong(act, i):
    """Nhan hoat dong cua cua so thu i theo quy tac 80% o muc 4.4.

    Cua so i ung voi giay [i*2, i*2+8), tuc chi so 4 Hz [i*8, i*8+32).
    Tra ve ID hoat dong neu mot nhan chiem it nhat 80% thoi luong, nguoc lai tra ve HON_HOP.
    """
    lo = i * (STEP // (FS // FS_ACT))        # i*128 mau 64 Hz -> i*8 mau 4 Hz
    seg = act[lo: lo + WIN // (FS // FS_ACT)]
    if len(seg) == 0:
        return HON_HOP
    gt, dem = np.unique(seg, return_counts=True)
    k = int(np.argmax(dem))
    return int(gt[k]) if dem[k] >= NGUONG_HD * len(seg) else HON_HOP


# --------------------------------------------------------------- chi muc cua so
def chi_muc(sid, can_acc=False):
    """Bang cua so cua mot doi tuong, kem co du dieu kien theo muc 2.2.

    Cua so bi loai khi: co gia tri khong huu han, hoac sigma_O <= 1e-6.
    sigma_O phu thuoc mat na, nen de bang thong ke tai lap duoc, buoc sang loc nay
    dung mat na danh gia co dinh 192..319 cho moi tap. Khi huan luyen, mat na sinh lai
    moi epoch nhung s_O luon bi chan duoi boi 1e-6 nen phep chia khong bao gio vo nghia.
    """
    d = nap_doi_tuong(sid, can_acc=can_acc)
    bvp, n_win = d['bvp'], so_cua_so(len(d['bvp']))
    M = mat_na(A_EVAL)
    O = M == 0
    huu_han = np.ones(n_win, dtype=bool)
    sigma = np.zeros(n_win, dtype=np.float32)
    act_win = np.zeros(n_win, dtype=np.int16)
    for i in range(n_win):
        r = cua_so(bvp, i)
        huu_han[i] = bool(np.all(np.isfinite(r)))
        if huu_han[i]:
            q = r[O]
            sigma[i] = float(np.sqrt(((q - q.mean()) ** 2).mean()))
        act_win[i] = nhan_hoat_dong(d['act'], i)
    du_dk = huu_han & (sigma > EPS_SIGMA)
    d.update(n_win=n_win, huu_han=huu_han, sigma=sigma, act_win=act_win, du_dk=du_dk)
    return d


def thong_ke(ds=range(1, 16)):
    """Bang thong ke theo doi tuong, phuc vu dieu kien hoan thanh cua Tuan 1."""
    RES.mkdir(exist_ok=True)
    hang, tong = [], dict(n_win=0, du_dk=0, loai_vo_han=0, loai_sigma=0)
    for sid in ds:
        d = chi_muc(sid)
        loai_vh = int(np.sum(~d['huu_han']))
        loai_sg = int(np.sum(d['huu_han'] & (d['sigma'] <= EPS_SIGMA)))
        r = dict(sid=sid, n_mau=len(d['bvp']), n_win=d['n_win'], du_dk=int(d['du_dk'].sum()),
                 loai_vo_han=loai_vh, loai_sigma=loai_sg,
                 hr_min=round(float(d['hr'].min()), 1), hr_max=round(float(d['hr'].max()), 1),
                 hon_hop=int(np.sum(d['act_win'] == HON_HOP)))
        for k in tong:
            tong[k] += r[k]
        hang.append(r)
        print('S%-3d mau=%7d  cua_so=%5d  du_dk=%5d  loai: vo_han=%d sigma=%d  hon_hop=%4d  HR[%.1f,%.1f]'
              % (sid, r['n_mau'], r['n_win'], r['du_dk'], loai_vh, loai_sg, r['hon_hop'],
                 r['hr_min'], r['hr_max']))

    import csv
    f = RES / 'thongke_dulieu.csv'
    with open(f, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(hang[0]))
        w.writeheader()
        w.writerows(hang)
    print('\nTong cua so truoc loai:', tong['n_win'], '| du dieu kien:', tong['du_dk'],
          '| loai vo han:', tong['loai_vo_han'], '| loai sigma:', tong['loai_sigma'])
    for ten, ss in SPLITS.items():
        print('  %-6s %-28s cua_so=%6d  du_dk=%6d' % (
            ten, 'S' + ',S'.join(map(str, ss)),
            sum(h['n_win'] for h in hang if h['sid'] in ss),
            sum(h['du_dk'] for h in hang if h['sid'] in ss)))
    print('Da ghi:', f)
    return hang


def xuat_mat_na_danh_gia(ds=range(1, 16)):
    """Tep vi tri khoang thieu dung khi danh gia (san pham ban giao o muc 5.2).

    Danh gia chinh dung mat na co dinh nen tep nay xac dinh boi mot hang so,
    van xuat ra de nguoi cham kiem tra duoc.
    """
    RES.mkdir(exist_ok=True)
    f = RES / 'matna_danhgia.json'
    f.write_text(json.dumps({
        'quy_tac': 'khoang thieu co dinh cho validation va test, muc 2.3 de cuong',
        'a': A_EVAL, 'b': A_EVAL + GAP - 1, 'do_dai': GAP, 'win': WIN,
        'huan_luyen': f'a ngau nhien deu trong {A_MIN}..{A_MAX}, sinh lai moi epoch, seed ghi trong cau hinh',
        'bo_vi_tri_khao_sat': [64, 128, 192, 256, 320],
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Da ghi:', f)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--thongke', action='store_true')
    a = ap.parse_args()
    if a.thongke:
        thong_ke()
        xuat_mat_na_danh_gia()
    else:
        # kiem tra nhanh cac bat bien quan trong
        d = chi_muc(1)
        rng = np.random.default_rng(42)
        r = cua_so(d['bvp'], 100)
        M = mat_na(a_ngau_nhien(rng))
        z, z_in, mu, s, sg = chuan_hoa(r, M)
        u = dau_vao(z_in, M)
        z_out = ghep_z(np.zeros(WIN, dtype=np.float32), z, M)
        r_out = ve_don_vi_goc(np.zeros(WIN, dtype=np.float32), r, M, mu, s)
        print('u', u.shape, '| mau thieu trong z_in deu bang 0:', bool(np.all(z_in[M == 1] == 0)))
        print('phan quan sat cua z_out trung z:', bool(np.allclose(z_out[M == 0], z[M == 0])))
        print('phan quan sat cua r_out trung r:', bool(np.allclose(r_out[M == 0], r[M == 0])))
        print('so fold:', len(FOLDS), '| fold 1 train:', fold_train(FOLDS[0]))
