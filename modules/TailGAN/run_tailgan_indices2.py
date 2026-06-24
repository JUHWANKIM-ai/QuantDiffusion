"""
원본 TailGAN.py 를 '수정 없이' indices2 데이터로 실행하는 runner.
- 원본 .py 파일들은 그대로. 여기서 모듈 전역(your_path 등)과 opt 만 주입한다.
- 먼저 prep_indices2_data.py 로 data/TailGAN/gan_data/indices2_6d/*.csv 가 만들어져 있어야 한다.

실행:  python modules/TailGAN/run_tailgan_indices2.py [n_epochs]
  n_epochs 인자를 주면 그 값으로(테스트용 짧게), 없으면 원본 기본값(3000) 사용.
"""
import os
import sys
from os.path import join

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TG_DIR = join(PROJECT, "modules", "TailGAN")
YOUR_PATH = join(PROJECT, "data", "TailGAN")
TICKERS = ["DJI", "NASDAQ", "JPM", "HSI", "GOLD", "WTI_OIL"]
DATA_NAME = "indices2_6d"
N_ROWS, N_COLS = 6, 256
# 다른 벤치마크와 동일하게 최종 모델 weight 는 result/.../synthetic_data/tailgan 에 저장
RESULT_TAILGAN = join(PROJECT, "result", "statistics", "synthetic_data", "tailgan")

# TailGAN 의 평면 import (from Dataset import ...) 가 되도록 경로/CWD 세팅
sys.path.insert(0, TG_DIR)
os.chdir(TG_DIR)
sys.argv = ["TailGAN.py"]            # argparse 가 default 를 쓰도록 argv 스텁

n_windows = len([f for f in os.listdir(join(YOUR_PATH, "gan_data", DATA_NAME)) if f.endswith(".csv")])
N_EPOCHS = int(os.environ.get("TG_EPOCHS", "3000"))   # 테스트: TG_EPOCHS=2 환경변수로 짧게

# ---- 1) 정적 포트폴리오 transition matrix 생성 (6종목) ----
import gen_static_port as gsp
gsp.your_path = YOUR_PATH
gsp.trans_parent_data_path = join(YOUR_PATH, "Static_Port_Transform")
os.makedirs(gsp.trans_parent_data_path, exist_ok=True)
gsp.Gen_StaticPort("LShort", n_stocks=N_ROWS, n_ports=50)

# ---- 2) gen_thresholds / Transform 경로 주입 ----
import gen_thresholds as gt
gt.your_path = YOUR_PATH
gt.parent_data_path = join(YOUR_PATH, "gan_data")

import Transform as tf
tf.your_path = YOUR_PATH
tf.trans_parent_data_path = join(YOUR_PATH, "Static_Port_Transform")

# ---- 3) TailGAN import + opt/경로 주입 (원본 로직 그대로) ----
import TailGAN as TG
TG.your_path = YOUR_PATH
o = TG.opt
o.n_rows, o.n_cols = N_ROWS, N_COLS
o.tickers = TICKERS
o.data_name = DATA_NAME
o.numNN = 1
o.n_epochs = N_EPOCHS
o.batch_size = int(os.environ.get("TG_BATCH", str(o.batch_size)))
# 원본 Train 의 DataLoader 는 drop_last 가 없어 마지막 부분배치가 Discriminator(Linear(batch))와 안 맞는다.
# -> len 을 batch_size 의 배수로 잘라 부분배치를 없앤다.
o.len = (n_windows // o.batch_size) * o.batch_size
# opt 변경 반영 (모듈 전역 재계산)
TG.R_shape = (o.n_rows, o.n_cols)
TG.PNL_shape = TG.Infer_Shape(TG.R_shape)
TG.gen_data_path = join(YOUR_PATH, "Gens", DATA_NAME)
TG.model_path = join(YOUR_PATH, "Models", DATA_NAME)
os.makedirs(TG.gen_data_path, exist_ok=True)
os.makedirs(TG.model_path, exist_ok=True)

print(f"[runner] your_path={YOUR_PATH}")
print(f"[runner] R_shape={TG.R_shape}, PNL_shape={TG.PNL_shape}, len={o.len}, numNN={o.numNN}, n_epochs={o.n_epochs}")
print(f"[runner] noise={o.noise_name}, lr_D={o.lr_D}, lr_G={o.lr_G}, batch={o.batch_size}, strategies={o.strategies}")

# import 시 옛 your_path='your_path' 로 생긴 빈 폴더 정리
import shutil
shutil.rmtree(join(TG_DIR, "your_path"), ignore_errors=True)

# ---- 디스크 절약: 매 epoch 덤프되는 Fake_id*.npy(분석용, 학습엔 불필요)는 저장 스킵 ----
# 원본 .py 는 그대로 두고 np.save 만 가로챔. (loss_id*.npy 는 그대로 저장됨)
import numpy as _np
_orig_np_save = _np.save
def _np_save_skip_fake(file, arr, *a, **k):
    if "Fake_id" in str(file):
        return
    return _orig_np_save(file, arr, *a, **k)
_np.save = _np_save_skip_fake

# ---- 최종 모델만 저장: generator 는 단일 파일(generator_final.pt)로 덮어쓰기, discriminator/중간본은 스킵 ----
# 원본은 100 epoch 마다 generator/discriminator 를 저장 -> torch.save 를 가로채
# generator 저장만 generator_final.pt 로 모아(덮어쓰기), 나머지는 저장 안 함. 학습 종료 시 = 최종 epoch weight.
import torch as _torch
os.makedirs(RESULT_TAILGAN, exist_ok=True)
GEN_FINAL = join(RESULT_TAILGAN, "generator_best.pt")
_orig_torch_save = _torch.save
def _torch_save_final_gen(obj, f, *a, **k):
    if "generator" in str(f):
        return _orig_torch_save(obj, GEN_FINAL, *a, **k)
    return None  # discriminator 등은 저장 안 함 (생성에 불필요)
_torch.save = _torch_save_final_gen

# ---- 4) 학습 (원본 Train; Screen_Ensemble 은 원본 버그라 생략) ----
TG.Train(o)

_np.save = _orig_np_save
_torch.save = _orig_torch_save
print(f"[runner] 최종 generator weight 저장: {GEN_FINAL}")
print("[runner] TailGAN 학습 완료. 생성물:", TG.gen_data_path)
