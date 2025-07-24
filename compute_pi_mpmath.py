import json
import os
import time
import pickle
from multiprocessing import Pool, cpu_count
from mpmath import mp, mpf, sqrt

# === 配置参数 ===
SAVE_INTERVAL_SECONDS = 300           # 保存周期（秒）
PRECISION = 5_000_000                 # π 小数精度（位数）
PROGRESS_FILE = "progress.json"       # 当前项数文件
SUM_FILE = "pi_sum.pkl"               # 当前总和文件（pickle）
PI_VALUE_FILE = "pi_value.txt"        # 当前 π 值文件
CORES = max(cpu_count() - 1, 1)       # 自动获取CPU核心数，至少1核
TERMS_PER_BATCH = 900                 # 每轮计算总项数（必须能被CORES整除）

# 校验 batch 是否合适
if TERMS_PER_BATCH % CORES != 0:
    raise ValueError("TERMS_PER_BATCH 必须能被 CORES 整除！")

# 设置高精度
mp.dps = PRECISION + 100

# 常量
L = 13591409
X = 640320
X3 = X ** 3
C = 426880 * sqrt(mpf(10005))  # Chudnovsky 常量

# === 使用递推计算 Chudnovsky 单项和区间和 ===
def chudnovsky_term_recursive(start_k, end_k):
    total = mpf(0)
    a = mpf(1)

    # 预热到 start_k 项
    for i in range(start_k):
        numerator = -(6*i + 1)*(2*i + 1)*(6*i + 5)
        denominator = ((i + 1)**3) * X3
        a *= mpf(numerator) / mpf(denominator)

    # 正式开始累加
    for i in range(start_k, end_k):
        term = a * (L + 545140134 * i)
        total += term
        numerator = -(6*i + 1)*(2*i + 1)*(6*i + 5)
        denominator = ((i + 1)**3) * X3
        a *= mpf(numerator) / mpf(denominator)

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
    return chudnovsky_term_recursive(start_k, start_k + batch_size)

# === 主计算函数 ===
def compute_pi():
    k, total = get_saved_progress()
    print(f"▶ 从第 {k} 项开始，目标精度 {PRECISION} 位，使用 {CORES} 核心")
    print(f"🔄 已恢复进度：k = {k}，当前总和估值略大于 π ≈ {str(C / total)[:14] if k else '未知'}")

    last_save = time.time()
    batch_size = TERMS_PER_BATCH // CORES

    try:
        with Pool(CORES) as pool:
            while True:
                batches = [(k + i * batch_size, batch_size) for i in range(CORES)]
                results = pool.starmap(compute_batch, batches)
                batch_sum = sum(results)
                total += batch_sum
                k += TERMS_PER_BATCH

                if time.time() - last_save >= SAVE_INTERVAL_SECONDS:
                    pi_val = C / total
                    pi_preview = str(pi_val)[:14]

                    print(f"[{time.strftime('%H:%M:%S')}] 已计算 {k} 项，π ≈ {pi_preview}")

                    # 保存 π 值（文本）
                    with open(PI_VALUE_FILE, "w") as f:
                        f.write(mp.nstr(pi_val, PRECISION))

                    # 保存进度
                    with open(PROGRESS_FILE, "w") as f:
                        json.dump({"k": k}, f)

                    # 保存总和
                    with open(SUM_FILE, "wb") as f:
                        pickle.dump(total, f)

                    last_save = time.time()

    except KeyboardInterrupt:
        print("\n🛑 用户中断，正在保存最后进度...")
        with open(PROGRESS_FILE, "w") as f:
            json.dump({"k": k}, f)
        with open(SUM_FILE, "wb") as f:
            pickle.dump(total, f)
        print("✅ 已保存退出，建议稍后继续计算。")

if __name__ == "__main__":
    compute_pi()
