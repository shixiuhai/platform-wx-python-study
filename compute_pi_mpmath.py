import json
import os
import time
from multiprocessing import Pool
import pickle
from mpmath import mp, mpf, fac, sqrt

# === 配置参数 ===
SAVE_INTERVAL_SECONDS = 30          # 保存周期（秒）
PRECISION = 10_000_000              # π 小数精度（位数）
PROGRESS_FILE = "progress.json"     # 当前项数文件
SUM_FILE = "pi_sum.pkl"             # 当前总和文件（改为 .pkl）
PI_VALUE_FILE = "pi_value.txt"      # 当前 π 值文件
CORES = 3                           # 使用核心数
TERMS_PER_BATCH = 10                # 每轮计算项数

# 设置高精度
mp.dps = PRECISION + 100  # 多预留位数，避免精度损失

# Chudnovsky 常数系数
C = 426880 * sqrt(mpf(10005))

# === 单项计算 ===
def chudnovsky_term(k):
    k = mpf(k)
    numerator = (-1) ** int(k) * fac(6 * int(k)) * (13591409 + 545140134 * k)
    denominator = fac(3 * int(k)) * (fac(int(k)) ** 3) * (640320 ** (3 * k))
    return numerator / denominator

# === 读取保存进度 ===
def get_saved_progress():
    k = 0
    total = mpf(0)
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            k = json.load(f).get("k", 0)
    if os.path.exists(SUM_FILE):
        with open(SUM_FILE, "rb") as f:
            total = pickle.load(f)
    return k, total

# === 批量计算多个项 ===
def compute_terms(start_k, end_k):
    return sum(chudnovsky_term(k) for k in range(start_k, end_k))

# === 主计算函数 ===
def compute_pi():
    k, total = get_saved_progress()
    print(f"▶ 从第 {k} 项开始，目标精度 {PRECISION} 位，使用 {CORES} 核心")

    last_save = time.time()

    while True:
        # 构造任务批次
        batches = [(k + i, k + i + 1) for i in range(TERMS_PER_BATCH)]

        # 多进程计算
        with Pool(CORES) as pool:
            results = pool.starmap(compute_terms, batches)
            batch_sum = sum(results)
            total += batch_sum

        k += TERMS_PER_BATCH

        # 是否保存
        if time.time() - last_save >= SAVE_INTERVAL_SECONDS:
            pi_val = C / total
            pi_short = str(pi_val)[:14]

            print(f"[{time.strftime('%H:%M:%S')}] 已计算 {k} 项，π ≈ {pi_short}")

            # 保存 pi（文本保存）
            with open(PI_VALUE_FILE, "w") as f:
                f.write(str(pi_val))

            # 保存进度（文本保存）
            with open(PROGRESS_FILE, "w") as f:
                json.dump({"k": k}, f)

            # 保存总和（改为二进制pickle保存）
            with open(SUM_FILE, "wb") as f:
                pickle.dump(total, f)

            last_save = time.time()

# === 程序入口 ===
if __name__ == "__main__":
    compute_pi()
