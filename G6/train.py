# -*- coding: utf-8 -*-
"""Vong huan luyen, muc 4.3 de cuong.

  - Adam, learning rate ban dau 1e-3, batch 128, toi da 100 epoch.
  - Early stopping patience 10 theo MAE validation GOP DEU GIUA DOI TUONG (CT 11), khoi phuc trong so tot nhat.
  - ReduceLROnPlateau factor 0.5, patience 5, theo cung chi so validation.
  - Seed chinh 42. Mat na huan luyen sinh lai moi lan lay mau, a deu trong 64..320 (muc 2.3).
  - Validation dung mat na co dinh 192..319.
  - Chuan hoa theo phan quan sat tung cua so, CT (3)(4). Khong co chuan hoa toan cuc.
  - Ghi lai phan cung, phien ban thu vien, thoi gian moi epoch (muc 4.3 va 4.5).

Ket qua: results/<id>.json (cau hinh, lich su, thoi gian) va weights/<id>.pt (trong so tot nhat).
Neu results/<id>.json da ton tai thi bo qua, de chay lai sau khi Colab ngat khong mat cong.

Chong mat cong khi Colab ngat giua chung: sau moi epoch, trang thai day du (mo hinh, optimizer,
scheduler, generator sinh mat na, lich su, so epoch cho) duoc ghi vao weights/<id>_ckpt.pt. Chay
lai dung lenh se tu doc file nay va tiep tuc dung epoch tiep theo, khong tinh lai tu dau. File
checkpoint bi xoa khi chay xong, khong anh huong ket qua cuoi (giong het chay lien mach mot lan).

Dung:
  python train.py --id e4_lam0.1 --lam 0.1                 chia co dinh S1-10 / S11-12
  python train.py --id e5_f1_lam0.1 --lam 0.1 --fold 1     fold 1 theo bang muc 2.4
  python train.py --id thu --lam 0.1 --epochs 2 --smoke    chay thu nhanh
"""
import argparse
import json
import platform
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

import data as D
from losses import loss_inpaint
from model import InpaintAE, dem_tham_so

HERE = Path(__file__).resolve().parent
RES, W = HERE / 'results', HERE / 'weights'


# ------------------------------------------------------------------ du lieu
def gom_cua_so(ds, can_acc=False):
    """Gom moi cua so du dieu kien cua cac doi tuong thanh tensor. Tra ve R (N,512), sid (N,), ACC (N,512,3) neu can."""
    Rs, sids, As = [], [], []
    for sid in ds:
        d = D.chi_muc(sid, can_acc=can_acc)
        idx = np.flatnonzero(d['du_dk'])
        R = np.stack([D.cua_so(d['bvp'], i) for i in idx])
        if can_acc:
            A = np.stack([D.acc_cua_so(d['acc'], i) for i in idx])
            As.append(A)
        Rs.append(R)
        sids.append(np.full(len(idx), sid))
    out = dict(R=torch.from_numpy(np.concatenate(Rs)).float(), sid=np.concatenate(sids))
    if can_acc:
        out['A'] = torch.from_numpy(np.concatenate(As)).float()
    return out


def chuan_hoa_torch(R, M):
    """CT (3)(4) tren ca lo, moi cua so mot mat na rieng. R, M: (B, 512)."""
    O = 1.0 - M
    n = O.sum(dim=1, keepdim=True)
    mu = (R * O).sum(dim=1, keepdim=True) / n
    sg = torch.sqrt((((R - mu) ** 2) * O).sum(dim=1, keepdim=True) / n)
    s = torch.clamp(sg, min=D.EPS_SIGMA)
    z = (R - mu) / s
    return z, z * O


def chuan_hoa_acc(A):
    """ACC quan sat day du trong cua so (gia dinh cua E2c), chuan hoa z-score tung truc tren ca 512 mau."""
    mu = A.mean(dim=1, keepdim=True)
    s = torch.clamp(A.std(dim=1, keepdim=True, unbiased=False), min=D.EPS_SIGMA)
    return (A - mu) / s


def mat_na_lo(a, device):
    """M (B, 512) tu vector a (B,)."""
    t = torch.arange(D.WIN, device=device).unsqueeze(0)
    a = a.unsqueeze(1)
    return ((t >= a) & (t < a + D.GAP)).float()


def lam_dau_vao(R, a, A=None):
    M = mat_na_lo(a, R.device)
    z, z_in = chuan_hoa_torch(R, M)
    kenh = [z_in, M]
    if A is not None:
        kenh += [chuan_hoa_acc(A)[:, :, c] for c in range(3)]
    return torch.stack(kenh, dim=2), z, M          # u (B,512,C), z (B,512), M (B,512)


# ------------------------------------------------------------------ danh gia
@torch.no_grad()
def mae_val_macro(model, val, device, batch=512, a_eval=(D.A_EVAL,)):
    """MAE tren khoang thieu co dinh, gop deu giua doi tuong theo CT (11).
    a_eval mac dinh chi la (192,) nhu giao thuc chinh. Bien the cai tien do tre truyen nhieu vi tri: lay trung binh
    deu MAE macro cua tung vi tri."""
    model.eval()
    R, sid, A = val['R'], val['sid'], val.get('A')
    ket = []
    for a0 in a_eval:
        mae = np.empty(len(R), dtype=np.float64)
        a = torch.full((batch,), a0, device=device)
        for lo in range(0, len(R), batch):
            Rb = R[lo:lo + batch].to(device)
            Ab = A[lo:lo + batch].to(device) if A is not None else None
            u, z, M = lam_dau_vao(Rb, a[:len(Rb)], Ab)
            zh = model(u)
            mae[lo:lo + batch] = ((zh - z).abs() * M).sum(dim=1).div(D.GAP).cpu().numpy()
        ket.append({int(s): float(mae[sid == s].mean()) for s in np.unique(sid)})
    ng = {s: float(np.mean([k[s] for k in ket])) for s in ket[0]}
    return float(np.mean(list(ng.values()))), ng


# ------------------------------------------------------------------ huan luyen
def huan_luyen(a):
    RES.mkdir(exist_ok=True); W.mkdir(exist_ok=True)
    f_json, f_pt, f_ckpt = RES / f'{a.id}.json', W / f'{a.id}.pt', W / f'{a.id}_ckpt.pt'
    if f_json.exists() and not a.ghi_de:
        print(f'Da co {f_json.name}, bo qua. Dung --ghi_de neu muon chay lai.')
        return json.loads(f_json.read_text(encoding='utf-8'))
    if a.ghi_de and f_ckpt.exists():
        f_ckpt.unlink()             # ghi de that su: bo qua checkpoint cu, khong phai chi la tiep tuc

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if a.fold:
        f = D.FOLDS[a.fold - 1]
        ds_train, ds_val = D.fold_train(f), f['val']
    else:
        ds_train, ds_val = D.SPLITS['train'], D.SPLITS['val']
    print(f'[{a.id}] device={device} | train S{ds_train} | val S{ds_val} | lam={a.lam} | lstm={not a.khong_lstm} | acc={a.acc}')

    t0 = time.time()
    tr, va = gom_cua_so(ds_train, a.acc), gom_cua_so(ds_val, a.acc)
    if a.smoke:
        tr = {k: v[:1024] for k, v in tr.items()}
        va = {k: v[:512] for k, v in va.items()}
    giay_nap = time.time() - t0
    print(f'  nap xong: train {len(tr["R"])} cua so, val {len(va["R"])} cua so ({giay_nap:.0f}s)')

    in_ch = 5 if a.acc else 2
    model = InpaintAE(in_ch, dung_lstm=not a.khong_lstm).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    gen = torch.Generator(device='cpu').manual_seed(a.seed)      # sinh vi tri mat na, tai lap duoc

    R_tr, A_tr = tr['R'], tr.get('A')
    N = len(R_tr)
    lich_su, tot_nhat, epoch_tot, cho = [], float('inf'), -1, 0
    trong_so_tot = deepcopy(model.state_dict())
    ep_bat_dau, giay_da_dung = 1, 0.0

    if f_ckpt.exists():
        try:
            ck = torch.load(f_ckpt, map_location=device)
            model.load_state_dict(ck['model'])
            opt.load_state_dict(ck['opt'])
            sched.load_state_dict(ck['sched'])
            # gen la Generator CPU rieng (sinh vi tri mat na), map_location=device o tren keo ca
            # gen_state sang GPU neu device=cuda, phai dua ve CPU truoc khi set_state cho generator nay
            gen.set_state(ck['gen_state'].cpu())
            trong_so_tot = ck['trong_so_tot']
            lich_su, tot_nhat, epoch_tot, cho = ck['lich_su'], ck['tot_nhat'], ck['epoch_tot'], ck['cho']
            ep_bat_dau, giay_da_dung = ck['epoch'] + 1, ck['giay_da_dung']
            print(f'  tiep tuc tu checkpoint: epoch {ep_bat_dau}, da chay {giay_da_dung/60:.1f} phut truoc do')
        except Exception as loi:
            # checkpoint hong (vi du bi ngat dung luc dang ghi, hoac Drive dong bo do dang): khong
            # crash ca luot, coi nhu chua co checkpoint va chay lai tu epoch 1 thay vi mat trang lenh
            print(f'  CANH BAO: checkpoint hong ({loi!r}), bo qua va chay lai tu epoch 1.')
            model = InpaintAE(in_ch, dung_lstm=not a.khong_lstm).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=a.lr)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
            gen = torch.Generator(device='cpu').manual_seed(a.seed)
            trong_so_tot = deepcopy(model.state_dict())
            f_ckpt.unlink(missing_ok=True)

    t_bat_dau = time.time()

    for ep in range(ep_bat_dau, a.epochs + 1):
        model.train()
        t_ep = time.time()
        hoan_vi = torch.randperm(N, generator=gen)
        tong, tong_g, tong_b, n_b = 0.0, 0.0, 0.0, 0
        for lo in range(0, N, a.batch):
            idx = hoan_vi[lo:lo + a.batch]
            Rb = R_tr[idx].to(device, non_blocking=True)
            Ab = A_tr[idx].to(device, non_blocking=True) if A_tr is not None else None
            ab = torch.randint(D.A_MIN, a.a_max + 1, (len(idx),), generator=gen).to(device)
            u, z, M = lam_dau_vao(Rb, ab, Ab)
            zh = model(u)
            loss, lg, lb = loss_inpaint(zh, z, M, ab, a.lam)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tong += loss.item(); tong_g += lg.item(); tong_b += lb.item(); n_b += 1

        mae_v, mae_v_ng = mae_val_macro(model, va, device, a_eval=a.a_eval)
        sched.step(mae_v)
        lr_hien = opt.param_groups[0]['lr']
        dt = time.time() - t_ep
        lich_su.append(dict(epoch=ep, loss=tong / n_b, l_gap=tong_g / n_b, l_boundary=tong_b / n_b,
                            mae_val_macro=mae_v, mae_val_theo_doi_tuong=mae_v_ng, lr=lr_hien, giay=dt))
        nan = not np.isfinite(tong / n_b) or not np.isfinite(mae_v)
        print('  ep %3d  loss %.5f  L_gap %.5f  L_bnd %.5f  |  MAE_val(macro) %.5f  lr %.2e  %.1fs%s'
              % (ep, tong / n_b, tong_g / n_b, tong_b / n_b, mae_v, lr_hien, dt, '  NaN!' if nan else ''))
        if nan:
            print('  chi so NaN, dung.'); break
        if mae_v < tot_nhat - 1e-7:
            tot_nhat, epoch_tot, cho = mae_v, ep, 0
            trong_so_tot = deepcopy(model.state_dict())
        else:
            cho += 1

        # ghi ra file tam roi doi ten: neu tien trinh bi giet giua chung luc dang ghi, file .tmp hong
        # nhung f_ckpt cu (con nguyen ven) khong bi dung, tranh checkpoint hong khong doc lai duoc
        f_tmp = f_ckpt.with_suffix('.tmp')
        torch.save(dict(epoch=ep, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        gen_state=gen.get_state(), trong_so_tot=trong_so_tot, lich_su=lich_su,
                        tot_nhat=tot_nhat, epoch_tot=epoch_tot, cho=cho,
                        giay_da_dung=giay_da_dung + (time.time() - t_bat_dau)), f_tmp)
        f_tmp.replace(f_ckpt)
        if cho >= a.patience:
            print(f'  early stopping: {a.patience} epoch khong cai thien.'); break

    model.load_state_dict(trong_so_tot)             # khoi phuc trong so tot nhat
    torch.save(trong_so_tot, f_pt)
    tong_giay = giay_da_dung + (time.time() - t_bat_dau)
    if f_ckpt.exists():
        f_ckpt.unlink()                              # chay xong, khong can checkpoint nua
    kq = dict(
        id=a.id, cau_hinh=dict(lam=a.lam, dung_lstm=not a.khong_lstm, acc=a.acc, fold=a.fold, seed=a.seed,
                               lr=a.lr, batch=a.batch, epochs_toi_da=a.epochs, patience=a.patience,
                               a_min=D.A_MIN, a_max=a.a_max,
                               a_eval=a.a_eval[0] if len(a.a_eval) == 1 else list(a.a_eval), smoke=a.smoke),
        doi_tuong=dict(train=ds_train, val=ds_val),
        so_cua_so=dict(train=int(N), val=int(len(va['R']))),
        tham_so=dem_tham_so(model),
        epoch_chay=len(lich_su), epoch_tot_nhat=epoch_tot, mae_val_tot_nhat=tot_nhat,
        mae_val_theo_doi_tuong=lich_su[epoch_tot - 1]['mae_val_theo_doi_tuong'] if epoch_tot > 0 else None,
        thoi_gian=dict(tong_giay=tong_giay, giay_moi_epoch=tong_giay / max(1, len(lich_su)),
                       nap_du_lieu_giay=giay_nap),
        moi_truong=dict(device=str(device),
                        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                        cpu=platform.processor(), threads=torch.get_num_threads(),
                        torch=torch.__version__, numpy=np.__version__, python=sys.version.split()[0]),
        lich_su=lich_su, trong_so=f_pt.name)
    f_json.write_text(json.dumps(kq, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'  xong: MAE_val tot nhat {tot_nhat:.5f} tai epoch {epoch_tot}/{len(lich_su)}, '
          f'{tong_giay/60:.1f} phut, {tong_giay/max(1,len(lich_su)):.1f} s/epoch. Ghi {f_json.name}, {f_pt.name}')
    return kq


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--id', required=True)
    ap.add_argument('--lam', type=float, required=True)
    ap.add_argument('--khong_lstm', action='store_true')
    ap.add_argument('--acc', action='store_true')
    ap.add_argument('--fold', type=int, default=0, help='1..5 theo bang muc 2.4; 0 = chia co dinh')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--smoke', action='store_true', help='chay thu tren 1024 cua so')
    ap.add_argument('--ghi_de', action='store_true')
    # Chi dung cho bien the cai tien do tre (ngoai de cuong). Mac dinh giu dung giao thuc chinh: a 64..320, danh gia o 192.
    ap.add_argument('--a_max', type=int, default=D.A_MAX, help='vi tri bat dau mat na lon nhat khi huan luyen (toi da 384)')
    ap.add_argument('--a_eval', type=int, nargs='+', default=[D.A_EVAL], help='cac vi tri dung de dung som tren validation')
    huan_luyen(ap.parse_args())
