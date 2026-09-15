"""Measure raw CPU inference FPS of an exported K-FACE ONNX model.

Usage:
    python tools/benchmark.py --model kface_n.onnx --size 320 --threads 4 --iters 200
"""
import argparse
import time
import numpy as np
import onnxruntime as ort


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    return ap.parse_args()


def main():
    args = parse_args()
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(args.model, sess_options=so, providers=["CPUExecutionProvider"])

    out_names = [o.name for o in sess.get_outputs()]
    dummy = np.random.randn(1, 3, args.size, args.size).astype(np.float32)

    for _ in range(args.warmup):
        sess.run(out_names, {"input": dummy})

    t0 = time.perf_counter()
    for _ in range(args.iters):
        sess.run(out_names, {"input": dummy})
    dt = time.perf_counter() - t0

    fps = args.iters / dt
    print(f"input={args.size}x{args.size} threads={args.threads} "
          f"avg_latency={1000*dt/args.iters:.2f}ms fps={fps:.1f}")


if __name__ == "__main__":
    main()
