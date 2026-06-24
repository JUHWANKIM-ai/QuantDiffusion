"""
indices2.csv -> TailGAN 입력 형식(폴더의 윈도우별 CSV)으로 변환.
원본 TailGAN 의 Dataset_IS / gen_thresholds 가 읽는 형식과 동일:
  data_root/gan_data/{data_name}/{i}.csv  (각 파일: n_cols행 x n_assets열, 헤더=티커, index 없음, 값=로그수익률)
"""
import os
from os.path import join
import numpy as np
import pandas as pd

# --- 설정 (3_benchmark 와 동일 데이터/분할) ---
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # quantDiffusion/
YOUR_PATH = join(PROJECT, "data", "TailGAN")        # = opt.your_path
DATA_NAME = "indices2_6d"
N_COLS = 256                                         # 윈도우 길이 (= seq_len)
TRAIN_RATIO = 0.7
STRIDE = 1


def main():
    df = pd.read_csv(join(PROJECT, "data", "indices2.csv"))
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df = df.set_index("Date").apply(pd.to_numeric).astype(float)
    tickers = list(df.columns)                       # ['DJI','NASDAQ','JPM','HSI','GOLD','WTI_OIL']

    log_ret = np.log1p(df.pct_change()).dropna()     # (T, 6) 로그수익률
    train_size = int(TRAIN_RATIO * len(log_ret))
    train = log_ret.iloc[:train_size]                # 학습 구간만 (test 누수 방지)

    out_dir = join(YOUR_PATH, "gan_data", DATA_NAME)
    os.makedirs(out_dir, exist_ok=True)
    # 기존 csv 정리 (재실행 대비)
    for f in os.listdir(out_dir):
        if f.endswith(".csv"):
            os.remove(join(out_dir, f))

    vals = train.values                              # (Ttrain, 6)
    n_windows = (len(vals) - N_COLS) // STRIDE + 1
    for i in range(n_windows):
        s = i * STRIDE
        window = vals[s:s + N_COLS]                  # (256, 6)
        pd.DataFrame(window, columns=tickers).to_csv(
            join(out_dir, "%d.csv" % (i + 1)), index=False
        )

    print(f"tickers      : {tickers}")
    print(f"data_name    : {DATA_NAME}")
    print(f"your_path    : {YOUR_PATH}")
    print(f"n_cols       : {N_COLS}")
    print(f"#windows(len): {n_windows}  -> {out_dir}")
    # 샘플 1개 형식 확인
    chk = pd.read_csv(join(out_dir, "1.csv"))
    print(f"sample csv   : shape={chk.shape} (행=시간, 열=자산), cols={list(chk.columns)}")
    print(f"  [tickers].values.T -> {chk[tickers].values.T.shape}  (자산, 시간) = TailGAN R_shape (n_rows, n_cols)")


if __name__ == "__main__":
    main()
