import json
import os
import time
from multiprocessing import Pool, cpu_count
import pickle
from mpmath import mp, mpf, sqrt

# === 配置参数 ===
SAVE_INTERVAL_SECONDS = 60          # 保存周期（秒）
PRECISION = 1_000_000               # π 小数精度（位数）
PROGRESS_FILE = "progress.json"     # 当前项数文件
SUM_FILE = "pi_sum.pkl"             # 当前总和文件（pickle）
PI_VALUE_FILE = "pi_value.txt"      # 当前 π 值文件
CORES = max(cpu_count() - 1, 1)      # 自动获取CPU核心数，至少1核
TERMS_PER_BATCH = 900               # 每轮计算总项数（必须能被CORES整除）

# 校验 TERMS_PER_BATCH 是否能被 CORES 整除
if TERMS_PER_BATCH % CORES != 0:
    raise ValueError("TERMS_PER_BATCH 必须能被 CORES 整除！")

# 设置高精度
mp.dps = PRECISION + 100  # 多预留位数，避免精度损失

# Chudnovsky 常数系数
C = 426880 * sqrt(mpf(10005))

# === 使用递推计算 Chudnovsky 单项和区间和 ===
def chudnovsky_term_recursive(start_k, end_k):
    """
    使用递推方式计算区间[start_k, end_k)的Chudnovsky项和。
    递推关系：
    a_0 = 1
    a_(k+1) = a_k * (- (6k+1)(2k+1)(6k+5)) / ((k+1)^3 * 640320^3)
    每项为 a_k * (13591409 + 545140134k)
    """
    total = mpf(0)
    k = mpf(start_k)
    # 初始化递推参数
    # 先计算a_0，再递推至start_k
    a = mpf(1)
    for i in range(int(start_k)):
        i_mpf = mpf(i)
        numerator = -(6*i_mpf + 1)*(2*i_mpf + 1)*(6*i_mpf + 5)
        denominator = ((i_mpf + 1)**3) * (640320**3)
        a = a * numerator / denominator

    for i in range(int(start_k), int(end_k)):
        k = mpf(i)
        term = a * (13591409 + 545140134 * k)
        total += term
        # 递推下一项a
        numerator = -(6*k + 1)*(2*k + 1)*(6*k + 5)
        denominator = ((k + 1)**3) * (640320**3)
        a = a * numerator / denominator

    return total

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

# === 计算单个批次 ===
def compute_batch(start_k, batch_size):
    # 计算[start_k, start_k + batch_size)的Chudnovsky项和
    return chudnovsky_term_recursive(start_k, start_k + batch_size)

# === 主计算函数 ===
def compute_pi():
    k, total = get_saved_progress()
    print(f"▶ 从第 {k} 项开始，目标精度 {PRECISION} 位，使用 {CORES} 核心")

    last_save = time.time()
    batch_size = TERMS_PER_BATCH // CORES

    with Pool(CORES) as pool:
        while True:
            # 构造每核计算区间
            batches = [(k + i * batch_size, batch_size) for i in range(CORES)]

            # 并行计算
            results = pool.starmap(compute_batch, batches)

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
