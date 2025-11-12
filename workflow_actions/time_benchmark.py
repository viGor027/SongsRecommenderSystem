import time
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_used = device.type == "cuda"
torch.manual_seed(0)


def timeit(fn):
    def inner(*args, **kwargs):
        if gpu_used:
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        out = fn(*args, **kwargs)
        if gpu_used:
            torch.cuda.synchronize()
        return time.perf_counter() - start_time, out

    return inner


@timeit
def bench(dtype, n=2048, reps=100):
    a = torch.randn(n, n, dtype=dtype, device=device)
    b = torch.randn(n, n, dtype=dtype, device=device)
    with torch.no_grad():
        for _ in range(5):
            a = a * b
        for _ in range(reps):
            a = a @ b
    return a.numel() * reps


def main():
    print(f"GPU użyte: {gpu_used}")
    dtypes = [
        ("float64", torch.float64),
        ("float32", torch.float32),
        ("bfloat16", torch.bfloat16),
        ("float16", torch.float16),
    ]

    for name, data_type in dtypes:
        t, n_multiplications = bench(data_type, n=2_000, reps=20)
        print(
            f"- {name:8s}: {t:.3f} s | mnożeń: {n_multiplications:,} | ~{n_multiplications / t:,.0f} mul/s"
        )


if __name__ == "__main__":
    main()
