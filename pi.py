import json
import os
import time
from decimal import Decimal, getcontext
from multiprocessing import Pool

# === 配置参数 ===
SAVE_INTERVAL_SECONDS = 30         # 每隔多少秒保存和打印一次
PRECISION = 1000000                # π 精度（小数点后位数）
PROGRESS_FILE = "progress.json"    # 保存当前进度（第几项）
SUM_FILE = "pi_sum.txt"            # 保存当前累加和（Chudnovsky总和）
PI_VALUE_FILE = "pi_value.txt"     # ✅ 保存当前 π 值
CORES = 3                          # 固定使用 3 个核心

# 设置 decimal 精度环境
getcontext().prec = PRECISION + 100  # 提高精度避免误差

# π 的常数系数
C = 426880 * Decimal(10005).sqrt()

# === 单项计算（Chudnovsky 公式） ===
def chudnovsky_term(k):
    from math import factorial
    k = Decimal(k)
    numerator = Decimal((-1) ** int(k)) * factorial(6 * int(k)) * (13591409 + 545140134 * k)
    denominator = factorial(3 * int(k)) * (factorial(int(k)) ** 3) * (640320 ** (3 * k))
    return numerator / denominator

# === 批量计算多个项 ===
def compute_terms(start_k, end_k):
    return sum(chudnovsky_term(k) for k in range(start_k, end_k))

# === 读取保存的进度 ===
def get_saved_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f).get("k", 0)
    return 0

# === 主计算函数 ===
def compute_pi(start_k=0, terms_per_batch=10, cores=CORES):
    k = start_k
    total = Decimal(0)

    # 恢复之前的总和（用于断点续算）
    if os.path.exists(SUM_FILE):
        with open(SUM_FILE, "r") as f:
            total = Decimal(f.read().strip())

    last_save = time.time()

    while True:
        # 准备任务批次
        batches = [(k + i, k + i + 1) for i in range(terms_per_batch)]

        # 多进程并行计算
        with Pool(cores) as pool:
            results = pool.starmap(compute_terms, batches)
            batch_sum = sum(results)
            total += batch_sum

        k += terms_per_batch

        # 每隔 SAVE_INTERVAL_SECONDS 秒保存 & 打印
        if time.time() - last_save >= SAVE_INTERVAL_SECONDS:
            pi_val = C / total

            # 控制台输出
            print(f"[{time.strftime('%H:%M:%S')}] 已计算项数: {k}, 当前 π 值 (前 12 位): {str(pi_val)[:14]}")

            # ✅ 保存 π 值到文件（每次覆盖）
            with open(PI_VALUE_FILE, "w") as f:
                f.write(str(pi_val))

            # 保存当前项数
            with open(PROGRESS_FILE, "w") as f:
                json.dump({"k": k}, f)

            # 保存累加和
            with open(SUM_FILE, "w") as f:
                f.write(str(total))

            last_save = time.time()

# === 启动入口 ===
if __name__ == "__main__":
    start_k = get_saved_progress()
    print(f"▶ 从第 {start_k} 项继续计算 π，使用 {CORES} 核心...")
    compute_pi(start_k=start_k)
