# -*- coding: utf-8 -*-
"""Kiem thu bat buoc muc 5.3 de cuong va kiem thu chong ro ri muc 5 phieu nhan xet.

Danh sach o muc 5.3:
  1. khong trung doi tuong giua cac tap
  2. doi rieng gia tri dich trong vung thieu khong lam thay doi dau vao
  3. so mau mat dung 128
  4. dau ra dung kich thuoc
  5. phan quan sat duoc giu nguyen
  6. chi so chi tinh dung vung quy dinh
  7. nhan HR can dung cua so
  8. cac doi chuan khong doc phan bi che

Kiem thu muc 5 phieu nhan xet: hai cua so co cung mau quan sat va cung mat na, khac nhau toan bo
gia tri that trong vung thieu, thi moi dau vao, thong ke chuan hoa va ket qua doi chuan phai giu
nguyen. Chi dich va gia tri loss/metric duoc phep thay doi.

Dung: python tests_leak.py
"""
import sys

import numpy as np

import baselines as B
import data as D
import hr_reader as H

RNG = np.random.default_rng(2026)
ket_qua = []


def kiem(ten, dieu_kien, ghi_chu=''):
    ket_qua.append((ten, bool(dieu_kien), ghi_chu))
    print('  [%s] %s%s' % ('OK  ' if dieu_kien else 'FAIL', ten, ('  | ' + ghi_chu) if ghi_chu else ''))
    return bool(dieu_kien)


def cau_hinh_bo_doc():
    return H.tai_cau_hinh() if H.CAU_HINH.exists() else dict(distance_mau=21, prominence_ty_le=0.1)


# ------------------------------------------------------------------ 1
def t1_doi_tuong_khong_trung():
    print('\n1. Khong trung doi tuong giua cac tap')
    tr, va, te = (set(D.SPLITS[k]) for k in ('train', 'val', 'test'))
    kiem('chia co dinh: train, val, test doi mot roi nhau',
         not (tr & va) and not (tr & te) and not (va & te))
    kiem('chia co dinh phu du 15 doi tuong', tr | va | te == set(range(1, 16)))
    ok_fold = True
    for k, f in enumerate(D.FOLDS, 1):
        t, v, h = set(f['test']), set(f['val']), set(D.fold_train(f))
        roi = not (t & v) and not (t & h) and not (v & h)
        du = (t | v | h) == set(range(1, 16))
        cd = len(t) == 3 and len(v) == 2 and len(h) == 10
        ok_fold &= roi and du and cd
    kiem('5 fold: moi fold roi nhau, du 15 nguoi, co cau 3/2/10', ok_fold)
    moi_nguoi = [s for s in range(1, 16)
                 if sum(s in f['test'] for f in D.FOLDS) != 1]
    kiem('moi doi tuong lam test dung mot lan trong 5 fold', not moi_nguoi,
         '' if not moi_nguoi else f'sai o {moi_nguoi}')


# ------------------------------------------------------------------ 2, 8
def t2_doi_dich_khong_doi_dau_vao(n=60):
    print('\n2 va 8. Doi gia tri that trong vung thieu, dau vao va doi chuan phai khong doi')
    d = D.chi_muc(1)
    ch = cau_hinh_bo_doc()
    sai_dau_vao = sai_thong_ke = sai_doi_chuan = 0
    dich_co_doi = 0
    for _ in range(n):
        i = int(RNG.integers(0, d['n_win']))
        a = D.a_ngau_nhien(RNG)
        M = D.mat_na(a)
        r1 = np.array(D.cua_so(d['bvp'], i), dtype=np.float32, copy=True)
        r2 = np.array(r1, copy=True)
        # thay toan bo gia tri that trong vung thieu bang mot tap khac han
        r2[M == 1] = RNG.normal(loc=1e3, scale=50, size=int(M.sum())).astype(np.float32)

        z1, zin1, mu1, s1, _ = D.chuan_hoa(r1, M)
        z2, zin2, mu2, s2, _ = D.chuan_hoa(r2, M)
        sai_dau_vao += int(not np.array_equal(D.dau_vao(zin1, M), D.dau_vao(zin2, M)))
        sai_thong_ke += int(not (mu1 == mu2 and s1 == s2))
        dich_co_doi += int(not np.allclose(z1[M == 1], z2[M == 1]))
        for ten, f in B.DOI_CHUAN.items():
            y1, _ = f(zin1, M, ch)
            y2, _ = f(zin2, M, ch)
            sai_doi_chuan += int(not np.allclose(y1, y2, atol=0, rtol=0))

    kiem('dau vao u = [z_in, M] khong doi', sai_dau_vao == 0, f'{sai_dau_vao}/{n} sai')
    kiem('thong ke chuan hoa mu va s khong doi', sai_thong_ke == 0, f'{sai_thong_ke}/{n} sai')
    kiem('ca ba doi chuan cho ket qua khong doi', sai_doi_chuan == 0, f'{sai_doi_chuan}/{3*n} sai')
    kiem('dich tai tao CO thay doi (kiem tra nguoc, chung to phep thu co hieu luc)',
         dich_co_doi == n, f'{dich_co_doi}/{n} doi')


# ------------------------------------------------------------------ 3
def t3_so_mau_mat(n=2000):
    print('\n3. So mau mat dung 128 va lien tuc')
    loi = 0
    for _ in range(n):
        a = D.a_ngau_nhien(RNG)
        M = D.mat_na(a)
        idx = np.flatnonzero(M == 1)
        loi += int(M.sum() != D.GAP or len(idx) != D.GAP
                   or idx[-1] - idx[0] + 1 != D.GAP
                   or a < D.A_MIN or a > D.A_MAX
                   or idx[0] < 64 or (D.WIN - 1 - idx[-1]) < 64)
    kiem(f'mat na huan luyen: dung {D.GAP} mau, lien tuc, moi ben con it nhat 64 mau', loi == 0,
         f'{loi}/{n} sai')
    Me = D.mat_na(D.A_EVAL)
    kiem('mat na danh gia co dinh dung 192..319',
         Me.sum() == D.GAP and np.flatnonzero(Me == 1)[0] == 192 and np.flatnonzero(Me == 1)[-1] == 319)
    kiem('spline luon du 64 moc moi ben voi moi a hop le',
         D.A_MIN - B.N_MOC_SPLINE >= 0 and D.A_MAX + D.GAP - 1 + B.N_MOC_SPLINE <= D.WIN - 1,
         f'a nho nhat {D.A_MIN}, b lon nhat {D.A_MAX + D.GAP - 1}')


# ------------------------------------------------------------------ 5
def t5_phan_quan_sat_giu_nguyen(n=60):
    print('\n5. Phan quan sat duoc giu nguyen')
    d = D.chi_muc(1)
    ch = cau_hinh_bo_doc()
    sai_ghep = sai_goc = sai_dc = 0
    for _ in range(n):
        i = int(RNG.integers(0, d['n_win']))
        M = D.mat_na(D.a_ngau_nhien(RNG))
        r = D.cua_so(d['bvp'], i)
        z, z_in, mu, s, _ = D.chuan_hoa(r, M)
        gia_dinh = RNG.normal(size=D.WIN).astype(np.float32)      # du doan bat ky
        z_out = D.ghep_z(gia_dinh, z, M)
        r_out = D.ve_don_vi_goc(gia_dinh, r, M, mu, s)
        sai_ghep += int(not np.allclose(z_out[M == 0], z[M == 0]))
        sai_goc += int(not np.allclose(r_out[M == 0], r[M == 0]))
        for ten, f in B.DOI_CHUAN.items():
            y, _ = f(z_in, M, ch)
            sai_dc += int(not np.allclose(y[M == 0], z_in[M == 0]))
    kiem('cong thuc (5): z_out trung z tai moi vi tri M=0', sai_ghep == 0)
    kiem('cong thuc (6): r_out trung r tai moi vi tri M=0', sai_goc == 0)
    kiem('doi chuan khong sua phan da biet', sai_dc == 0, f'{sai_dc}/{3*n} sai')


# ------------------------------------------------------------------ 7
def t7_nhan_hr_can_dung():
    print('\n7. Nhan HR can dung cua so')
    lech, sai_act = [], []
    for sid in range(1, 16):
        d = D.nap_doi_tuong(sid)
        n_win = D.so_cua_so(len(d['bvp']))
        if len(d['hr']) != n_win:
            lech.append(sid)
        # cua so cuoi cung phai con du nhan hoat dong 4 Hz
        can = (n_win - 1) * (D.STEP // (D.FS // D.FS_ACT)) + D.WIN // (D.FS // D.FS_ACT)
        if len(d['act']) < can:
            sai_act.append(sid)
    kiem('len(label) = so cua so tinh theo cong thuc (2) o ca 15 doi tuong', not lech,
         '' if not lech else f'lech o {lech}')
    kiem('nhan hoat dong 4 Hz du dai cho cua so cuoi cung', not sai_act,
         '' if not sai_act else f'thieu o {sai_act}')
    # cua so i ung voi giay [i*2, i*2+8)
    kiem('cua so i bat dau tai giay i*2 o ca hai truc thoi gian',
         D.STEP / D.FS == 2.0 and (D.STEP // (D.FS // D.FS_ACT)) / D.FS_ACT == 2.0)


# ------------------------------------------------------------------ 4
def t4_dau_ra_dung_kich_thuoc():
    print('\n4. Dau ra dung kich thuoc')
    import torch
    from model import InpaintAE, dem_tham_so
    import train as T
    for ten, c, l, n_mong in (('day du', 2, True, 117_233), ('bo Bi-LSTM', 2, False, 36_337), ('them ACC', 5, True, 117_569)):
        m = InpaintAE(c, l).eval()
        with torch.no_grad():
            y = m(torch.zeros(7, D.WIN, c))
        kiem(f'{ten}: dau ra (B, 512) va {n_mong} tham so',
             tuple(y.shape) == (7, D.WIN) and dem_tham_so(m) == n_mong, f'ra {tuple(y.shape)}, {dem_tham_so(m)} tham so')
    # duong ong huan luyen: u co dung 2 (hoac 5) kenh, M trong u trung mat na, z_in bang 0 trong gap
    R = torch.randn(5, D.WIN)
    a = torch.tensor([64, 100, 192, 300, 320])
    u, z, M = T.lam_dau_vao(R, a)
    kiem('u = [z_in, M] co dang (B, 512, 2)', tuple(u.shape) == (5, D.WIN, 2))
    kiem('kenh M trong u trung mat na va moi hang dung 128 mau', torch.equal(u[:, :, 1], M) and bool((M.sum(1) == D.GAP).all()))
    kiem('kenh z_in bang 0 tai moi vi tri M=1', bool((u[:, :, 0][M == 1] == 0).all()))
    # chuan hoa torch trung chuan hoa numpy cua data.py
    rn, Mn = R[2].numpy(), M[2].numpy()
    z_np, zin_np, mu, s, _ = D.chuan_hoa(rn, Mn)
    kiem('chuan hoa torch trung chuan hoa numpy (CT 3, 4)',
         np.allclose(z[2].numpy(), z_np, atol=1e-5) and np.allclose(u[2, :, 0].numpy(), zin_np, atol=1e-5))


# ------------------------------------------------------------------ 6
def t6_chi_so_dung_vung():
    print('\n6. Chi so chi tinh dung vung quy dinh')
    import torch
    import evaluate as E
    from losses import l_gap, l_boundary
    M = D.mat_na(D.A_EVAL)
    Z = RNG.normal(size=(20, D.WIN))
    Y = np.array(Z, copy=True)
    Y[:, M == 0] += 1e3                       # pha hong hoan toan phan quan sat
    mae, rmse = E.chi_so_gap(Z, Y, M)
    kiem('MAE va RMSE (CT 10) khong doi khi chi pha phan quan sat', np.allclose(mae, 0) and np.allclose(rmse, 0))
    Y2 = np.array(Z, copy=True)
    Y2[:, M == 1] += 2.0                      # lech dung 2 trong gap
    mae2, rmse2 = E.chi_so_gap(Z, Y2, M)
    kiem('lech hang so 2 trong gap cho MAE = RMSE = 2', np.allclose(mae2, 2) and np.allclose(rmse2, 2))
    # gop macro (CT 11): hai nguoi, nguoi A 3 cua so, nguoi B 1 cua so -> trung binh deu giua nguoi
    mac, ng = E.gop_macro([1, 1, 1, 5], [1, 1, 1, 2])
    kiem('gop macro (CT 11) la trung binh deu giua doi tuong, khong theo so cua so', abs(mac - 3.0) < 1e-9, f'ra {mac}')
    # loss: L_gap khong doi khi sua ngoai gap; L_boundary chi nhin 4 cap quanh mep
    zt, Mt = torch.tensor(Z, dtype=torch.float32), torch.tensor(M).unsqueeze(0).repeat(20, 1)
    at = torch.full((20,), D.A_EVAL)
    yt = torch.tensor(Y, dtype=torch.float32)
    kiem('L_gap (CT 7) bang 0 khi chi sai ngoai gap', float(l_gap(yt, zt, Mt)) == 0.0)
    y3 = zt.clone(); y3[:, D.A_EVAL + 10: D.A_EVAL + 100] += 5.0     # sai sau trong gap, xa 4 cap mep
    kiem('L_boundary (CT 8, 9) bang 0 khi sai nam xa 4 cap quanh mep', float(l_boundary(y3, zt, Mt, at)) == 0.0)


if __name__ == '__main__':
    print('KIEM THU BAT BUOC (muc 5.3) VA CHONG RO RI (muc 5 phieu nhan xet)')
    t1_doi_tuong_khong_trung()
    t2_doi_dich_khong_doi_dau_vao()
    t3_so_mau_mat()
    t5_phan_quan_sat_giu_nguyen()
    t7_nhan_hr_can_dung()
    t4_dau_ra_dung_kich_thuoc()
    t6_chi_so_dung_vung()
    hong = [t for t, ok, _ in ket_qua if not ok]
    print('\n%d/%d kiem thu dat.' % (len(ket_qua) - len(hong), len(ket_qua)))
    if hong:
        print('Khong dat:', hong)
        sys.exit(1)
