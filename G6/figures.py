# -*- coding: utf-8 -*-
"""Sinh hinh cho bao cao tu tep ket qua, khong dien so thu cong (muc 5.1, 5.2).

Quy tac chon hinh vi du (co dinh truoc, muc 5.2): tren tap test S13-S15, lay cua so co MAE
cua mo hinh chinh gan nhat voi phan vi 5 (tot), 50 (trung vi), 95 (kem). Khong chon hinh dep.
Net ve dung nhieu kieu duong (lien, dut, cham) de doc duoc khi in den trang.
Nhan trong hinh viet tieng Viet co dau vi hinh dat vao bao cao.

Dung: python figures.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import data as D
from evaluate import doc_khoa

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / 'results', HERE / 'figs'
PHAN_VI = {'tốt': 5, 'trung vị': 50, 'kém': 95}
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 10})
PHAY = matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'.replace('.', ','))   # so thap phan kieu Viet

# Mau co dinh theo tung phuong phap, dung chung moi hinh de doc xuyen suot (bo mau tham chieu cua skill dataviz:
# xanh duong, cam, xanh ngoc la ba slot dau; xam trung tinh cho phuong phap kem nhat). Van giu net lien/dut/cham
# va dau diem de in den trang van doc duoc.
MAU = dict(mo_hinh='#2a78d6', lap='#eb6834', tuyen='#1baf7a', spline='#6b6a65', bien_the='#9ec5f4',
           that='#1f1f1d', vung='#e9e8e3')
MAU_LAMBDA = ['#86b6ef', '#5598e7', '#256abf', '#104281']                          # lambda 0 -> 1, nhat -> dam


def phay(x, n=3):
    return f'{x:.{n}f}'.replace('.', ',')


def truc_phay(*axes):
    for a in axes:
        a.yaxis.set_major_formatter(PHAY)


def doc_bang(ten):
    with open(RES / f'{ten}_theo_cuaso.csv', encoding='utf-8-sig') as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def hinh_vi_du(khoa):
    chinh = f'mo_hinh_chinh[{khoa["mo_hinh_chinh"]}]'
    b = doc_bang('e1_test')
    gap = np.load(RES / 'e1_test_dudoan_gap.npz')
    mae = b[f'mae_{chinh}'].astype(float)
    M = D.mat_na(D.A_EVAL)
    t = np.arange(D.WIN) / D.FS
    fig, ax = plt.subplots(3, 1, figsize=(7.2, 6.6), sharex=True)
    chon = {}
    for k, (nhan, pv) in enumerate(PHAN_VI.items()):
        muc = np.percentile(mae, pv)
        j = int(np.argmin(np.abs(mae - muc)))
        sid, win = int(b['sid'][j]), int(b['win'][j])
        r = np.asarray(D.cua_so(D.nap_doi_tuong(sid)['bvp'], win), dtype=np.float64)
        z, _, _, _, _ = D.chuan_hoa(r, M)
        g = M == 1
        a = ax[k]
        a.axvspan(t[g][0], t[g][-1], color=MAU['vung'], lw=0)
        a.plot(t, z, color=MAU['that'], lw=1.0, label='Tín hiệu thật')
        a.plot(t[g], gap[chinh][j], color=MAU['mo_hinh'], lw=1.7, ls='--', label='Mô hình chính')
        a.plot(t[g], gap['lap_chu_ky'][j], color=MAU['lap'], lw=1.5, ls=':', label='Lặp chu kỳ')
        a.set_ylabel('z')
        truc_phay(a)
        a.set_title('Trường hợp %s (phân vị %d): S%d, cửa sổ %d, MAE mô hình %s, lặp chu kỳ %s'
                    % (nhan, pv, sid, win, phay(mae[j]), phay(float(b['mae_lap_chu_ky'][j]))), fontsize=9)
        chon[nhan] = dict(phan_vi=pv, sid=sid, cua_so=win, mae_mo_hinh=round(float(mae[j]), 4),
                          mae_lap_chu_ky=round(float(b['mae_lap_chu_ky'][j]), 4))
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.0))
    ax[-1].set_xlabel('Thời gian (giây), vùng xám là đoạn bị che 2 giây')
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(FIG / 'hinh_vidu_tot_trungvi_kem.png', dpi=200)
    plt.close(fig)
    return chon


def hinh_so_sanh(khoa):
    kq = json.loads((RES / 'e1_test.json').read_text(encoding='utf-8'))
    ten = {'tuyen_tinh': 'Nội suy\ntuyến tính', 'spline': 'Spline\nbậc ba', 'lap_chu_ky': 'Lặp\nchu kỳ',
           f'e2a_bo_lstm[{khoa["e2a_bo_lstm"]}]': 'Bỏ\nBi-LSTM', f'e2b_lam0[{khoa["e2b_lam0"]}]': 'λ = 0',
           f'e2c_them_acc[{khoa["e2c_them_acc"]}]': 'Thêm\nACC', f'mo_hinh_chinh[{khoa["mo_hinh_chinh"]}]': 'Mô hình\nchính'}
    khoa_pp = list(ten)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for a, (m, nhan) in zip(ax, (('mae_macro', 'MAE vùng thiếu (z)'), ('mae_hr_pres', 'MAE_HR,pres (bpm)'))):
        v = [kq[k][m] for k in khoa_pp]
        mau = [MAU['tuyen'], MAU['spline'], MAU['lap'], MAU['bien_the'], MAU['bien_the'], MAU['bien_the'], MAU['mo_hinh']]
        bars = a.bar(range(len(v)), v, color=mau, edgecolor='white', lw=0.8)
        for bi, vi in zip(bars, v):
            a.text(bi.get_x() + bi.get_width() / 2, vi, f'{vi:.2f}'.replace('.', ','), ha='center', va='bottom',
                   fontsize=7)
        a.set_xticks(range(len(v)))
        a.set_xticklabels([ten[k] for k in khoa_pp], fontsize=7)
        a.set_title(nhan, fontsize=9)
        truc_phay(a)
    fig.tight_layout()
    fig.savefig(FIG / 'hinh_sosanh_phuongphap.png', dpi=200)
    plt.close(fig)


def hinh_e4():
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    kieu = ['-', '--', '-.', ':']
    for k, lam in enumerate(('0', '0.1', '0.5', '1')):
        d = json.loads((RES / f'e4_lam{lam}.json').read_text(encoding='utf-8'))
        ep = [h['epoch'] for h in d['lich_su']]
        v = [h['mae_val_macro'] for h in d['lich_su']]
        tot = f'{d["mae_val_tot_nhat"]:.4f}'.replace('.', ',')
        ax.plot(ep, v, color=MAU_LAMBDA[k], ls=kieu[k], lw=1.5, label=f'λ = {lam.replace(".", ",")} (tốt nhất {tot})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE validation (gộp đều theo đối tượng)')
    truc_phay(ax)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / 'hinh_e4_lambda.png', dpi=200)
    plt.close(fig)


def hinh_e3(khoa):
    kq = json.loads((RES / 'e1_test.json').read_text(encoding='utf-8'))['e3_theo_hoat_dong']
    chinh = f'mo_hinh_chinh[{khoa["mo_hinh_chinh"]}]'
    ten = {'hon hop/chuyen tiep': 'hỗn hợp', 'chuyen tiep': 'chuyển tiếp', 'ngoi': 'ngồi',
           'cau thang': 'cầu thang', 'bi lac': 'bi lắc', 'dap xe': 'đạp xe', 'lai xe': 'lái xe',
           'an trua': 'ăn trưa', 'di bo': 'đi bộ', 'lam viec': 'làm việc'}
    nhom = [n for n in kq if kq[n]['so_cua_so'] > 0]
    x = np.arange(len(nhom))
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.bar(x - 0.2, [kq[n]['lap_chu_ky']['mae_macro'] for n in nhom], 0.4, color=MAU['lap'], edgecolor='white', lw=0.8,
           label='Lặp chu kỳ')
    ax.bar(x + 0.2, [kq[n][chinh]['mae_macro'] for n in nhom], 0.4, color=MAU['mo_hinh'], edgecolor='white', lw=0.8,
           label='Mô hình chính')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{ten.get(n, n)}\n(n={kq[n]["so_cua_so"]})' for n in nhom], fontsize=7)
    ax.set_ylabel('MAE vùng thiếu (z)')
    truc_phay(ax)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / 'hinh_e3_hoatdong.png', dpi=200)
    plt.close(fig)


def hinh_e5():
    """MAE ngoai mau cua du 15 doi tuong tu kiem dinh cheo 5 fold (chi ve khi da co e5_tonghop.json)."""
    f = RES / 'e5_tonghop.json'
    if not f.exists():
        print('Chua co e5_tonghop.json, bo qua hinh E5')
        return
    kq = json.loads(f.read_text(encoding='utf-8'))['theo_doi_tuong']
    ds = sorted(kq, key=int)
    x = np.arange(len(ds))
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.bar(x - 0.2, [kq[s]['lap_chu_ky'] for s in ds], 0.4, color=MAU['lap'], edgecolor='white', lw=0.8,
           label='Lặp chu kỳ')
    ax.bar(x + 0.2, [kq[s]['mo_hinh'] for s in ds], 0.4, color=MAU['mo_hinh'], edgecolor='white', lw=0.8,
           label='Mô hình (λ chọn trong từng fold)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{s}' for s in ds], fontsize=8)
    ax.set_ylabel('MAE vùng thiếu (z), ngoài mẫu')
    truc_phay(ax)
    ax.set_ylim(0, 1.3)                         # chua cho chu thich khong de len cot
    ax.legend(fontsize=8, frameon=False, ncol=2, loc='upper center')
    fig.tight_layout()
    fig.savefig(FIG / 'hinh_e5_15doituong.png', dpi=200)
    plt.close(fig)


def hinh_vi_tri():
    """Khao sat do nhay theo vi tri khoang thieu tren test (muc 2.3), chi ve khi da co khaosat_vitri.json."""
    f = RES / 'khaosat_vitri.json'
    if not f.exists():
        print('Chua co khaosat_vitri.json, bo qua hinh khao sat vi tri')
        return
    kq = json.loads(f.read_text(encoding='utf-8'))['vi_tri']
    khoa = doc_khoa()
    ds = sorted(kq, key=int)
    x = [int(a) for a in ds]
    ve = [('tuyen_tinh', 'Nội suy tuyến tính', '-', 'o', MAU['tuyen']), ('spline', 'Spline bậc ba', '--', 's', MAU['spline']),
          ('lap_chu_ky', 'Lặp chu kỳ', ':', '^', MAU['lap']),
          (f'mo_hinh_chinh[{khoa["mo_hinh_chinh"]}]', 'Mô hình chính', '-', 'D', MAU['mo_hinh'])]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for a, (m, nhan) in zip(ax, (('mae_macro', 'MAE vùng thiếu (z)'), ('mae_hr_pres', 'MAE_HR,pres (bpm)'))):
        for k, ten, ls, mk, mau in ve:
            a.plot(x, [kq[d]['phuong_phap'][k][m] for d in ds], ls=ls, marker=mk, color=mau, lw=1.2, ms=4,
                   label=ten)
        a.set_xticks(x)
        a.set_xlabel('Vị trí bắt đầu khoảng thiếu a (mẫu)')
        a.set_title(nhan, fontsize=9)
        truc_phay(a)
    ax[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / 'hinh_khaosat_vitri.png', dpi=200)
    plt.close(fig)


def hinh_cai_tien_do_tre(tap='test'):
    """Cai tien: sai so theo thoi gian cho sau doan mat (chi ve khi da co caitien_dotre.json)."""
    f = RES / 'caitien_dotre.json'
    if not f.exists():
        print('Chua co caitien_dotre.json, bo qua hinh cai tien')
        return
    kq = json.loads(f.read_text(encoding='utf-8'))['tap'][tap]
    khoa = doc_khoa()
    ds = sorted(kq, key=lambda a: kq[a]['vi_tri']['cho_sau_doan_mat_giay'])
    x = {a: kq[a]['vi_tri']['cho_sau_doan_mat_giay'] for a in ds}
    ve = [(f'mo_hinh_chinh[{khoa["mo_hinh_chinh"]}]', 'Mô hình chính', '-', 'D', MAU['mo_hinh']),
          ('them[caitien_amax384_lam0.5]', 'Biến thể độ trễ thấp', '--', 's', '#104281'),
          ('lap_chu_ky', 'Lặp chu kỳ', ':', '^', MAU['lap']),
          ('tuyen_tinh', 'Nội suy tuyến tính', '-', 'o', MAU['tuyen'])]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for a_, (m, nhan) in zip(ax, (('mae_macro', 'MAE vùng thiếu (z)'), ('mae_hr_pres', 'MAE_HR,pres (bpm)'))):
        a_.axvspan(-0.15, 0.85, color=MAU['vung'], lw=0, zorder=0)
        for k, ten, ls, mk, mau in ve:
            dd = [a for a in ds if k in kq[a]['phuong_phap']]
            if not dd:
                continue
            a_.plot([x[a] for a in dd], [kq[a]['phuong_phap'][k][m] for a in dd], ls=ls, marker=mk, color=mau, lw=1.4, ms=4,
                    label=ten, zorder=3)
        a_.set_xticks(sorted(set(x.values())))
        a_.xaxis.set_major_formatter(PHAY)
        a_.set_xlim(-0.15, 5.15)
        a_.set_xlabel('Thời gian chờ sau khi đoạn mất kết thúc (giây)')
        a_.set_title(nhan, fontsize=9)
        truc_phay(a_)
    for a_ in ax:                                    # chu thich vung xam o giua truc, khong de len duong ve
        a_.text(0.35, 0.6, 'ngoài vùng\nhuấn luyện\ncủa mô hình\nchính', ha='center', va='center', fontsize=6.3,
                color='0.35', transform=a_.get_xaxis_transform())
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / 'hinh_caitien_dotre.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    FIG.mkdir(exist_ok=True)
    hinh_e5()
    hinh_vi_tri()
    hinh_cai_tien_do_tre()
    khoa = doc_khoa()
    chon = hinh_vi_du(khoa)
    hinh_so_sanh(khoa)
    hinh_e4()
    hinh_e3(khoa)
    (RES / 'hinh_vidu_chon.json').write_text(json.dumps(dict(quy_tac='phan vi MAE mo hinh chinh tren test', chon=chon),
                                                      ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(chon, ensure_ascii=False, indent=2))
    print('Da ghi hinh vao', FIG)
