# -*- coding: utf-8 -*-
"""Kham pha du lieu (muc 2.3 cua bao cao). CHI dung train va validation (S1..S12), khong doc test S13..S15.

Sinh 5 hinh va results/khampha_dulieu.json:
  hinh_khampha_tinhieu.png    : BVP tho 8 giay theo hoat dong (cua so o vi tri trung vi cua hoat dong, S1)
  hinh_khampha_nhiptim.png    : phan bo nhip tim tu ECG cua tung nguoi
  hinh_khampha_hoatdong.png   : ty le cua so theo nhom hoat dong
  hinh_khampha_biendo.png     : do lech chuan cua so (don vi goc) theo nguoi va theo hoat dong
  hinh_khampha_photanso.png   : pho cong suat trung binh cua BVP, ngoi so voi di bo
Dung: python kham_pha_du_lieu.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

import data as D

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / 'results', HERE / 'figs'
DS = D.SPLITS['train'] + D.SPLITS['val']                         # S1..S12
assert set(DS).isdisjoint(D.SPLITS['test']), 'Khong duoc dung test S13..S15 trong buoc kham pha'
SEED = 42

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 10,
                     'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False})
MAU = dict(xanh='#2a78d6', cam='#eb6834', ngoc='#1baf7a', xam='#6b6a65', that='#1f1f1d', luoi='#e9e8e3')
PHAY = matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'.replace('.', ','))
TEN = {0: 'chuyển tiếp', 1: 'ngồi', 2: 'cầu thang', 3: 'bi lắc', 4: 'đạp xe', 5: 'lái xe', 6: 'ăn trưa', 7: 'đi bộ',
       8: 'làm việc', D.HON_HOP: 'hỗn hợp'}

# ------------------------------------------------------------------ nap du lieu (chi S1..S12)
du = {}
for sid in DS:
    d = D.chi_muc(sid)
    du[sid] = d
    print(f'S{sid}: {d["n_win"]} cua so')
act_win = {s: du[s]['act_win'] for s in DS}
sigma = {s: du[s]['sigma'] for s in DS}                         # do lech chuan phan quan sat (mat na 192..319), don vi goc
hr = {s: du[s]['hr'] for s in DS}
ket = dict(doi_tuong=DS, ghi_chu='Chi dung train va validation (S1..S12); khong dung test.')

# ------------------------------------------------------------------ hinh 1: tin hieu theo hoat dong
chon = [1, 7, 4, 2]                                              # ngoi, di bo, dap xe, cau thang
d1 = du[1]
fig, axs = plt.subplots(len(chon), 1, figsize=(6.4, 4.8), sharex=True)
t = np.arange(D.WIN) / D.FS
ket['cua_so_mau_S1'] = {}
for ax, a in zip(axs, chon):
    idx = np.flatnonzero((act_win[1] == a) & d1['du_dk'])
    i = int(idx[len(idx) // 2])                                   # cua so o vi tri trung vi cua hoat dong
    r = np.asarray(D.cua_so(d1['bvp'], i), dtype=np.float64)
    ax.plot(t, r, color=MAU['that'], lw=1.0)
    ax.set_ylabel('BVP')
    ax.set_title(f'{TEN[a].capitalize()} (S1, cửa sổ {i}, HR ECG {d1["hr"][i]:.1f} bpm)'.replace('.', ','), fontsize=9.5, loc='left')
    ket['cua_so_mau_S1'][TEN[a]] = dict(cua_so=i, hr_ecg=round(float(d1['hr'][i]), 1), do_lech_chuan=round(float(r.std()), 2))
axs[-1].set_xlabel('Thời gian trong cửa sổ (giây)')
fig.tight_layout(); fig.savefig(FIG / 'hinh_khampha_tinhieu.png', dpi=200); plt.close(fig)

# ------------------------------------------------------------------ hinh 2: nhip tim theo nguoi
fig, ax = plt.subplots(figsize=(6.4, 2.9))
dat = [hr[s][du[s]['du_dk']] for s in DS]
bp = ax.boxplot(dat, whis=(0, 100), patch_artist=True, widths=0.6, showfliers=False,
                medianprops=dict(color=MAU['that'], lw=1.5), whiskerprops=dict(color=MAU['xam']), capprops=dict(color=MAU['xam']))
for b in bp['boxes']:
    b.set(facecolor='#9ec5f4', edgecolor=MAU['xanh'])
ax.axhline(170, color=MAU['cam'], ls=':', lw=1.4)
ax.text(12.45, 172, 'trần 170 bpm của bộ đọc', color=MAU['cam'], ha='right', va='bottom', fontsize=9)
ax.set_xticklabels([f'S{s}' for s in DS]); ax.set_ylabel('Nhịp tim từ ECG (bpm)')
ax.set_ylim(30, 200); ax.yaxis.grid(True, color=MAU['luoi']); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(FIG / 'hinh_khampha_nhiptim.png', dpi=200); plt.close(fig)
ket['nhip_tim'] = {f'S{s}': dict(min=round(float(x.min()), 1), trung_vi=round(float(np.median(x)), 1), max=round(float(x.max()), 1))
                   for s, x in zip(DS, dat)}
ket['nhip_tim_tren_170'] = {f'S{s}': int((x > 170).sum()) for s, x in zip(DS, dat)}
tat_ca = np.concatenate(dat)
ket['nhip_tim_chung'] = dict(min=round(float(tat_ca.min()), 1), trung_vi=round(float(np.median(tat_ca)), 1), max=round(float(tat_ca.max()), 1),
                             ty_le_tren_170=round(float((tat_ca > 170).mean()), 5), so_cua_so=int(len(tat_ca)))

# ------------------------------------------------------------------ hinh 3: ty le cua so theo nhom hoat dong
dem = {}
for s in DS:
    m = du[s]['du_dk']
    for a, c in zip(*np.unique(act_win[s][m], return_counts=True)):
        dem[int(a)] = dem.get(int(a), 0) + int(c)
tong = sum(dem.values())
thu_tu = sorted(dem, key=lambda a: dem[a])
fig, ax = plt.subplots(figsize=(6.4, 2.9))
ax.barh([TEN[a] for a in thu_tu], [100 * dem[a] / tong for a in thu_tu], color=MAU['xanh'], height=0.6)
for k, a in enumerate(thu_tu):
    ax.text(100 * dem[a] / tong + 0.4, k, f'{100 * dem[a] / tong:.1f}'.replace('.', ',') + '% (' + f'{dem[a]:,}'.replace(',', '.') + ')',
            va='center', fontsize=9)
ax.set_xlabel('Tỷ lệ số cửa sổ (%)'); ax.set_xlim(0, 28); ax.xaxis.grid(True, color=MAU['luoi']); ax.set_axisbelow(True)
ax.xaxis.set_major_formatter(PHAY)
fig.tight_layout(); fig.savefig(FIG / 'hinh_khampha_hoatdong.png', dpi=200); plt.close(fig)
ket['hoat_dong'] = {TEN[a]: dict(so_cua_so=dem[a], ty_le=round(dem[a] / tong, 4)) for a in thu_tu[::-1]}
ket['hoat_dong_tong'] = tong
# so nguoi dong gop moi nhom
ket['hoat_dong_so_nguoi'] = {TEN[a]: int(sum(1 for s in DS if np.any(act_win[s][du[s]['du_dk']] == a))) for a in thu_tu[::-1]}

# ------------------------------------------------------------------ hinh 4: bien do theo nguoi va theo hoat dong
fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.0), gridspec_kw=dict(width_ratios=[1.15, 1]))
dsg = [sigma[s][du[s]['du_dk']] for s in DS]
b1 = axs[0].boxplot(dsg, whis=(5, 95), patch_artist=True, widths=0.6, showfliers=False, medianprops=dict(color=MAU['that'], lw=1.3))
for b in b1['boxes']:
    b.set(facecolor='#9ec5f4', edgecolor=MAU['xanh'])
axs[0].set_yscale('log'); axs[0].set_xticklabels([f'S{s}' for s in DS], rotation=90, fontsize=8.5)
axs[0].set_ylabel('Độ lệch chuẩn phần quan sát (đơn vị gốc)'); axs[0].set_title('(a) Theo người', fontsize=9.5, loc='left')
axs[0].yaxis.grid(True, color=MAU['luoi']); axs[0].set_axisbelow(True)
nhom = [a for a in (1, 8, 6, 5, 4, 7, 2, 3, 0, D.HON_HOP)]
dhd = [np.concatenate([sigma[s][du[s]['du_dk'] & (act_win[s] == a)] for s in DS]) for a in nhom]
b2 = axs[1].boxplot(dhd, whis=(5, 95), patch_artist=True, widths=0.6, showfliers=False, medianprops=dict(color=MAU['that'], lw=1.3))
for b in b2['boxes']:
    b.set(facecolor='#9ec5f4', edgecolor=MAU['xanh'])
axs[1].set_yscale('log'); axs[1].set_xticklabels([TEN[a] for a in nhom], rotation=90, fontsize=8.5)
axs[1].set_title('(b) Theo hoạt động', fontsize=9.5, loc='left'); axs[1].yaxis.grid(True, color=MAU['luoi']); axs[1].set_axisbelow(True)
fig.tight_layout(); fig.savefig(FIG / 'hinh_khampha_biendo.png', dpi=200); plt.close(fig)
med_nguoi = [float(np.median(x)) for x in dsg]
ket['bien_do'] = dict(trung_vi_theo_nguoi={f'S{s}': round(m, 1) for s, m in zip(DS, med_nguoi)},
                      ti_so_nguoi_lon_nho=round(max(med_nguoi) / min(med_nguoi), 2),
                      trung_vi_theo_hoat_dong={TEN[a]: round(float(np.median(x)), 1) for a, x in zip(nhom, dhd)},
                      ti_so_p95_p5_chung=round(float(np.percentile(np.concatenate(dsg), 95) / np.percentile(np.concatenate(dsg), 5)), 1))

# ------------------------------------------------------------------ hinh 5: pho cong suat
rng = np.random.default_rng(SEED)
N_MAU = 600
pho = {}
for a, nhan, mau in ((1, 'ngồi', MAU['xanh']), (7, 'đi bộ', MAU['cam'])):
    cs = []
    for s in DS:
        for i in np.flatnonzero((act_win[s] == a) & du[s]['du_dk']):
            cs.append((s, int(i)))
    pick = rng.choice(len(cs), size=min(N_MAU, len(cs)), replace=False)
    P = []
    for k in pick:
        s, i = cs[int(k)]
        r = np.asarray(D.cua_so(du[s]['bvp'], i), dtype=np.float64)
        f, p = welch(r - r.mean(), fs=D.FS, nperseg=D.WIN, window='hann')
        P.append(p / p.sum())
    pho[a] = (f, np.mean(P, axis=0), nhan, mau, len(P))
fig, ax = plt.subplots(figsize=(6.4, 2.9))
ax.axvspan(0.5, 8.0, color=MAU['luoi'], lw=0)
for a, (f, p, nhan, mau, n) in pho.items():
    ax.semilogy(f, p, color=mau, lw=1.6, label=f'{nhan} (n = {n})')
ax.set_xlim(0, 12); ax.set_ylim(1e-6, 1.0)
ax.text(4.25, 0.3, 'dải lọc 0,5 đến 8 Hz', ha='center', fontsize=9, color=MAU['xam'])
ax.set_xlabel('Tần số (Hz)'); ax.set_ylabel('Công suất tương đối'); ax.legend(frameon=False, loc='upper right')
ax.xaxis.set_major_formatter(PHAY)
fig.tight_layout(); fig.savefig(FIG / 'hinh_khampha_photanso.png', dpi=200); plt.close(fig)
ket['pho_tan_so'] = {}
for a, (f, p, nhan, mau, n) in pho.items():
    band = (f >= 0.5) & (f <= 8)
    dinh = f[band][int(np.argmax(p[band]))]
    ket['pho_tan_so'][nhan] = dict(n=n, tan_so_dinh_hz=round(float(dinh), 3), ty_le_cong_suat_0p5_8=round(float(p[band].sum() / p.sum()), 4),
                                   ty_le_cong_suat_duoi_0p5=round(float(p[f < 0.5].sum() / p.sum()), 4),
                                   ty_le_cong_suat_tren_8=round(float(p[f > 8].sum() / p.sum()), 4))

(RES / 'khampha_dulieu.json').write_text(json.dumps(ket, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: ket[k] for k in ('nhip_tim_chung', 'hoat_dong_tong', 'bien_do', 'pho_tan_so')}, ensure_ascii=False, indent=1))
print('Da ghi 5 hinh va results/khampha_dulieu.json')
