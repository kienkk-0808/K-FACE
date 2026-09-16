"""Author: kienkk

Measure CPU inference latency/FPS of one or more ONNX models with
identical session settings, interleaved across rounds so results are
comparable regardless of run order.

Usage:
    python tools/benchmark.py --model kface_n.onnx --size 320 --threads 4
    python tools/benchmark.py --model kface_n.onnx --model other_model.onnx --size 320
"""
import argparse
import time

import numpy as np
import onnxruntime as ort


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="repeatable")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--iters", type=int, default=200, help="runs per model per round")
    ap.add_argument("--rounds", type=int, default=5, help="interleaved rounds; best is kept")
    ap.add_argument("--warmup", type=int, default=50)
    return ap.parse_args()


def make_session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name


def main():
    args = parse_args()
    x = np.random.randn(1, 3, args.size, args.size).astype(np.float32)

    sessions = [make_session(p, args.threads) for p in args.model]
    for sess, name in sessions:
        for _ in range(args.warmup):
            sess.run(None, {name: x})

    best = [float("inf")] * len(sessions)
    for _ in range(args.rounds):
        for i, (sess, name) in enumerate(sessions):
            t0 = time.perf_counter()
            for _ in range(args.iters):
                sess.run(None, {name: x})
            best[i] = min(best[i], (time.perf_counter() - t0) / args.iters * 1000)

    print(f"input={args.size}x{args.size} threads={args.threads} "
          f"iters={args.iters} rounds={args.rounds} (interleaved, best round)")
    ref = best[-1]
    for path, ms in zip(args.model, best):
        print(f"  {path:50s} latency={ms:6.2f} ms  fps={1000 / ms:7.1f}  ({ms / ref:.2f}x last)")


if __name__ == "__main__":
    main()
