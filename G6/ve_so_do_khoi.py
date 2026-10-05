# -*- coding: utf-8 -*-
"""Ve lai so do khoi kien truc (Hinh 1 de cuong) voi ky hieu thong nhat z, z mu theo CT (5).
Bo cuc, cac khoi va kich thuoc tensor giu dung nhu so do da duyet trong de cuong; chi doi nhan dau ra.

Dung: python ve_so_do_khoi.py   -> figs/hinh_sodokhoi.png
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix'})

W, H = 1.7, 1.05          # kich thuoc moi khoi
X = [0.0, 1.95, 3.9, 5.85, 7.8]
Y1, Y2 = 2.2, 0.0


# Nen nhat theo nhom khoi (chu va vien van den): dau vao/ra xam, ma hoa xanh duong, nut co chai cam, giai ma xanh ngoc
NEN = dict(io='#ecebe6', ma_hoa='#d6e6fa', co_chai='#fbd9ca', giai_ma='#cdeee3')


def khoi(ax, x, y, tieu_de, dong2, kich_thuoc, nen='ma_hoa'):
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle='round,pad=0.0,rounding_size=0.06',
                                fc=NEN[nen], ec='black', lw=1.6))
    ax.text(x + W / 2, y + H * 0.78, tieu_de, ha='center', va='center', fontsize=12.5, fontweight='bold')
    ax.text(x + W / 2, y + H * 0.48, dong2, ha='center', va='center', fontsize=11.5)
    ax.text(x + W / 2, y + H * 0.18, kich_thuoc, ha='center', va='center', fontsize=12, family='monospace')


def mui_ten(ax, x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', lw=1.6, color='black',
                                                                   mutation_scale=16, shrinkA=0, shrinkB=0))


def ngoac(ax, x0, x1, y, nhan, tren=True):
    d = 0.1 if tren else -0.1
    ax.plot([x0, x0, x1, x1], [y, y + d, y + d, y], color='black', lw=1.2)
    ax.text((x0 + x1) / 2, y + d + (0.17 if tren else -0.17), nhan, ha='center', va='center',
            fontsize=12.5, style='italic')


fig, ax = plt.subplots(figsize=(10.5, 4.6))
khoi(ax, X[0], Y1, 'Đầu vào', 'PPG che + mặt nạ M', '(512, 2)', nen='io')
khoi(ax, X[1], Y1, 'Conv1D 16, k = 7', 'ReLU, MaxPool 2', '(256, 16)')
khoi(ax, X[2], Y1, 'Conv1D 32, k = 7', 'ReLU, MaxPool 2', '(128, 32)')
khoi(ax, X[3], Y1, 'Conv1D 64, k = 7', 'ReLU, MaxPool 2', '(64, 64)')
khoi(ax, X[4], Y1, 'Bi-LSTM', '64 ẩn mỗi chiều', '(64, 128)', nen='co_chai')
khoi(ax, X[0], Y2, 'Up 2, Conv1D 32', 'k = 7, ReLU', '(128, 32)', nen='giai_ma')
khoi(ax, X[1], Y2, 'Up 2, Conv1D 16', 'k = 7, ReLU', '(256, 16)', nen='giai_ma')
khoi(ax, X[2], Y2, 'Up 2, Conv1D 1', 'k = 7, tuyến tính', '(512, 1)', nen='giai_ma')
khoi(ax, X[3], Y2, 'Đầu ra', r'Ghép $z$ và $\hat{z}$ theo $M$', '(512, 1)', nen='io')

for i in range(4):
    mui_ten(ax, X[i] + W, Y1 + H / 2, X[i + 1], Y1 + H / 2)
for i in range(3):
    mui_ten(ax, X[i] + W, Y2 + H / 2, X[i + 1], Y2 + H / 2)
xm = X[4] + W / 2
ym = (Y1 + Y2 + H) / 2
ax.plot([xm, xm, X[0] + W / 2], [Y1, ym, ym], color='black', lw=1.6)
mui_ten(ax, X[0] + W / 2, ym, X[0] + W / 2, Y2 + H)

ngoac(ax, X[1], X[3] + W, Y1 + H + 0.08, 'Bộ mã hóa 1D-CNN')
ngoac(ax, X[4], X[4] + W, Y1 + H + 0.08, 'Nút cổ chai')
ngoac(ax, X[0], X[2] + W, Y2 - 0.08, 'Bộ giải mã 1D-CNN', tren=False)
ax.text(X[4] + W, Y1 + H + 0.62, 'Kích thước tensor ghi theo (bước thời gian, kênh)', ha='right',
        va='center', fontsize=11, style='italic')

ax.set_xlim(-0.1, X[4] + W + 0.1)
ax.set_ylim(Y2 - 0.55, Y1 + H + 0.8)
ax.set_aspect('equal')
ax.axis('off')
fig.tight_layout()
(HERE / 'figs').mkdir(exist_ok=True)
fig.savefig(HERE / 'figs' / 'hinh_sodokhoi.png', dpi=220)
print('Da ghi figs/hinh_sodokhoi.png')
