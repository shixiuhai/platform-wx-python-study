import json
import os
import time
from multiprocessing import Pool, cpu_count
import pickle
from mpmath import mp, mpf, fac, sqrt

# === 配置参数 ===
SAVE_INTERVAL_SECONDS = 120         # 保存周期（秒）
PRECISION = 1_000_000               # π 小数精度（位数）
PROGRESS_FILE = "progress.json"     # 当前项数文件
SUM_FILE = "pi_sum.pkl"             # 当前总和文件（改为 .pkl）
PI_VALUE_FILE = "pi_value.txt"      # 当前 π 值文件
CORES = cpu_count()-1                 # 自动获取CPU核心数
TERMS_PER_BATCH = 1000              # 每轮计算总项数（必须能被CORES整除）

# 校验 TERMS_PER_BATCH 是否能被 CORES 整除
if TERMS_PER_BATCH % CORES != 0:
    raise ValueError("TERMS_PER_BATCH 必须能被 CORES 整除！")

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
    # 计算从 start_k 到 end_k -1 项的和
    s = mpf(0)
    for i in range(start_k, end_k):
        s += chudnovsky_term(i)
    return s

# === 主计算函数 ===
def compute_pi():
    k, total = get_saved_progress()
    print(f"▶ 从第 {k} 项开始，目标精度 {PRECISION} 位，使用 {CORES} 核心")

    last_save = time.time()
    batch_size = TERMS_PER_BATCH // CORES

    with Pool(CORES) as pool:
        while True:
            batches = [(k + i * batch_size, k + (i + 1) * batch_size) for i in range(CORES)]

            results = pool.starmap(compute_terms, batches)
            batch_sum = sum(results)
            total += batch_sum

            k += TERMS_PER_BATCH

            # 定时保存
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

                # 保存总和（pickle二进制）
                with open(SUM_FILE, "wb") as f:
                    pickle.dump(total, f)

                last_save = time.time()

if __name__ == "__main__":
    compute_pi()
