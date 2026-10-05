# -*- coding: utf-8 -*-
"""Chay tron E5: 5 fold x 4 lambda (muc 2.4, luoi day du), roi danh gia ngoai mau.
Chay tach roi: python -u chay_e5.py > logs/e5.log. Moi luot tu bo qua neu da co ket qua,
nen neu bi ngat thi chay lai dung lenh nay la tiep tuc."""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def buoc(ten, cmd):
    print(f'[{datetime.now():%H:%M:%S}] BAT DAU {ten}: {" ".join(cmd[1:])}', flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=HERE)
    print(f'[{datetime.now():%H:%M:%S}] XONG {ten} ({(time.time()-t0)/60:.1f} phut)', flush=True)


buoc('E5 huan luyen', [PY, '-u', 'experiments.py', 'e5', '--luoi', '0', '0.1', '0.5', '1'])
buoc('E5 danh gia ngoai mau', [PY, '-u', 'evaluate.py', '--e5'])
print(f'[{datetime.now():%H:%M:%S}] HOAN TAT TOAN BO E5', flush=True)
