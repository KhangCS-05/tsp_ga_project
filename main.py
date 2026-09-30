"""
File chạy chính của dự án GA-TSP.

Cách dùng:
    python main.py <file.tsp> [tùy chọn]

Ví dụ:
    python main.py data/ch150.tsp --algorithm ga
    python main.py data/ch150.tsp --algorithm sa
    python main.py data/ch150.tsp --algorithm nn
    python main.py data/ch150.tsp --algorithm all
    python main.py data/ch150.tsp --algorithm all --plot

Tác giả: [Tên bạn]
Ngày: [Ngày hôm nay]
"""

import sys
import argparse
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.data_loader import load_tsp, print_problem_info
from core.distance import compute_distance_matrix
from core.ga import run_ga
from core.algorithms.nearest_neighbor import nearest_neighbor
from core.algorithms.simulated_annealing import simulated_annealing


# Đáp án tối ưu đã biết của một số bài toán TSPLIB
KNOWN_OPTIMA = {
    "berlin52": 7542,
    "eil51": 426,
    "ch150": 6528,
    "kroA100": 21282,
    "tsp225": 3919,
    "pr439": 107217,
    "pcb442": 50778,
}


def parse_args():
    """Phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Ứng dụng giải bài toán TSP bằng nhiều thuật toán",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "tsp_file", type=str,
        help="Đường dẫn tới file .tsp (định dạng TSPLIB)",
    )

    parser.add_argument(
        "--algorithm", type=str, default="ga",
        choices=["ga", "sa", "nn", "all"],
        help="Thuật toán: ga, sa, nn, hoặc all (chạy cả 3)",
    )

    # Tham số GA
    parser.add_argument("--pop-size", type=int, default=50,
                        help="Kích thước quần thể")
    parser.add_argument("--generations", type=int, default=500,
                        help="Số thế hệ")
    parser.add_argument("--mutation-rate", type=float, default=0.1,
                        help="Tỉ lệ đột biến")
    parser.add_argument("--elite-size", type=int, default=5,
                        help="Số cá thể elitism")
    parser.add_argument("--tournament-k", type=int, default=3,
                        help="Tournament size")
    parser.add_argument("--crossover", type=str, default="ox",
                        choices=["ox", "pmx"],
                        help="Loại lai ghép")
    parser.add_argument("--mutation", type=str, default="inversion",
                        choices=["swap", "inversion"],
                        help="Loại đột biến")
    parser.add_argument("--no-2opt", action="store_true",
                        help="Tắt 2-opt cho GA")
    parser.add_argument("--two-opt-iter", type=int, default=20,
                        help="Số vòng 2-opt cho GA")
    parser.add_argument("--two-opt-every", type=int, default=1,
                        help="Áp dụng 2-opt mỗi N thế hệ")

    # Tham số SA
    parser.add_argument("--sa-max-iter", type=int, default=100_000,
                        help="SA: số vòng lặp tối đa")
    parser.add_argument("--sa-initial-temp", type=float, default=1000.0,
                        help="SA: nhiệt độ ban đầu")
    parser.add_argument("--sa-cooling", type=float, default=0.9999,
                        help="SA: hệ số giảm nhiệt")
    parser.add_argument("--sa-local-every", type=int, default=1000,
                        help="SA: chạy 2-opt mỗi N vòng")
    parser.add_argument("--sa-local-iter", type=int, default=50,
                        help="SA: số vòng 2-opt mỗi lần")

    # Tham số output
    parser.add_argument("--seed", type=int, default=None,
                        help="Hạt giống ngẫu nhiên")
    parser.add_argument("--plot", action="store_true",
                        help="Vẽ biểu đồ")
    parser.add_argument("--save", type=str, default=None,
                        help="Lưu kết quả JSON")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Thư mục lưu kết quả")
    parser.add_argument("--quiet", action="store_true",
                        help="Không in tiến độ")

    return parser.parse_args()


def run_nn(dist, args, verbose=True):
    """Chạy Nearest Neighbor."""
    t0 = time.time()
    result = nearest_neighbor(dist, n_starts=None, verbose=False)
    elapsed = time.time() - t0
    return {
        "name": "NN",
        "best_tour": result.best_tour,
        "best_length": result.best_length,
        "time_seconds": elapsed,
        "history": [result.best_length],
    }


def run_sa(dist, args, verbose=True):
    """Chạy Simulated Annealing."""
    nn_result = nearest_neighbor(dist, n_starts=None, verbose=False)

    result = simulated_annealing(
        dist,
        initial_tour=nn_result.best_tour,
        max_iterations=args.sa_max_iter,
        initial_temp=args.sa_initial_temp,
        final_temp=0.01,
        cooling_rate=args.sa_cooling,
        move_type="inversion",
        local_search_every=args.sa_local_every,
        local_search_iter=args.sa_local_iter,
        seed=args.seed,
        verbose=verbose and not args.quiet,
    )
    return {
        "name": "SA",
        "best_tour": result.best_tour,
        "best_length": result.best_length,
        "time_seconds": result.time_seconds,
        "history": result.history,
    }


def run_ga_wrapper(dist, args, verbose=True):
    """Chạy Genetic Algorithm."""
    result = run_ga(
        dist=dist,
        pop_size=args.pop_size,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        elite_size=args.elite_size,
        tournament_k=args.tournament_k,
        crossover_type=args.crossover,
        mutation_type=args.mutation,
        use_2opt=not args.no_2opt,
        two_opt_iterations=args.two_opt_iter,
        two_opt_every=args.two_opt_every,
        seed=args.seed,
        verbose=verbose and not args.quiet,
    )
    return {
        "name": "GA",
        "best_tour": result.best_tour,
        "best_length": result.best_length,
        "time_seconds": result.time_seconds,
        "history": result.history,
    }


def print_result(result, optimal=None):
    """In kết quả một thuật toán."""
    print(f"\n--- {result['name']} ---")
    print(f"Best length:  {result['best_length']:.2f}")
    print(f"Thời gian:    {result['time_seconds']:.3f} giây")
    if optimal:
        gap = 100 * (result['best_length'] - optimal) / optimal
        print(f"Sai số:       {gap:.3f}%")


def print_comparison(results, optimal=None):
    """In bảng so sánh các thuật toán."""
    print()
    print("=" * 70)
    print("BẢNG SO SÁNH CÁC THUẬT TOÁN")
    print("=" * 70)
    print(f"{'Thuật toán':<12} {'Best':<14} {'Sai số':<12} {'Thời gian':<15}")
    print("-" * 70)

    for r in results:
        gap_str = "—"
        if optimal:
            gap = 100 * (r['best_length'] - optimal) / optimal
            gap_str = f"{gap:.3f}%"
        time_str = f"{r['time_seconds']:.3f}s"
        print(f"{r['name']:<12} {r['best_length']:<14.2f} {gap_str:<12} {time_str:<15}")

    print("=" * 70)

    # Tìm thuật toán tốt nhất
    if optimal:
        best = min(results, key=lambda r: r['best_length'])
        fastest = min(results, key=lambda r: r['time_seconds'])
        print(f"\nChất lượng tốt nhất: {best['name']} ({best['best_length']:.2f}, sai {100*(best['best_length']-optimal)/optimal:.3f}%)")
        print(f"Nhanh nhất:          {fastest['name']} ({fastest['time_seconds']:.3f}s)")


def main():
    """Hàm chính."""
    args = parse_args()

    print("=" * 70)
    print("  ỨNG DỤNG GIẢI BÀI TOÁN NGƯỜI DU LỊCH (TSP)")
    print("=" * 70)
    print()

    # 1. Đọc dữ liệu
    try:
        problem = load_tsp(args.tsp_file)
    except FileNotFoundError as e:
        print(f"LỖI: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"LỖI định dạng file: {e}")
        sys.exit(1)

    print_problem_info(problem)
    print()

    # 2. Tính ma trận khoảng cách
    print("Đang tính ma trận khoảng cách...")
    t0 = time.time()
    dist = compute_distance_matrix(problem.coords, problem.edge_weight_type)
    print(f"  → Xong trong {time.time() - t0:.3f} giây")
    print(f"  → Kích thước: {dist.shape}, dtype: {dist.dtype}")
    print(f"  → Bộ nhớ: {dist.nbytes / 1024:.1f} KB")
    print()

    # 3. Đáp án tối ưu
    optimal = KNOWN_OPTIMA.get(problem.name)
    if optimal:
        print(f"Đáp án tối ưu đã biết: {optimal}")
    else:
        print("Không có đáp án tối ưu để so sánh.")
    print()

    # 4. Chạy thuật toán
    results = []

    if args.algorithm == "all":
        print("=" * 70)
        print("CHẠY SO SÁNH 3 THUẬT TOÁN")
        print("=" * 70)

        print("\n[1/3] Nearest Neighbor...")
        results.append(run_nn(dist, args, verbose=False))
        print_result(results[-1], optimal)

        print("\n[2/3] Simulated Annealing...")
        results.append(run_sa(dist, args, verbose=not args.quiet))
        print_result(results[-1], optimal)

        print("\n[3/3] Genetic Algorithm...")
        results.append(run_ga_wrapper(dist, args, verbose=not args.quiet))
        print_result(results[-1], optimal)

        print_comparison(results, optimal)

    elif args.algorithm == "nn":
        print("Chạy Nearest Neighbor...")
        results.append(run_nn(dist, args, verbose=False))
        print_result(results[-1], optimal)

    elif args.algorithm == "sa":
        print("Chạy Simulated Annealing...")
        results.append(run_sa(dist, args, verbose=not args.quiet))
        print_result(results[-1], optimal)

    elif args.algorithm == "ga":
        print("Chạy Genetic Algorithm...")
        results.append(run_ga_wrapper(dist, args, verbose=not args.quiet))
        print_result(results[-1], optimal)

    # 5. Lưu JSON
    if args.save:
        output = {
            "problem": problem.name,
            "tsp_file": str(args.tsp_file),
            "optimal": optimal,
            "results": [
                {
                    "name": r["name"],
                    "best_length": float(r["best_length"]),
                    "time_seconds": float(r["time_seconds"]),
                    "best_tour": r["best_tour"].tolist(),
                    "history": r["history"][:1000],
                }
                for r in results
            ],
        }
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nĐã lưu kết quả JSON: {args.save}")

    # 6. Vẽ biểu đồ
    if args.plot:
        try:
            from core.visualize import (
                plot_convergence, plot_tour, plot_comparison,
            )
            import numpy as np

            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)

            print("\nĐang vẽ biểu đồ...")

            # Biểu đồ so sánh (chỉ vẽ khi có ít nhất 2 thuật toán có history dài)
            multi_hist = [r for r in results if len(r["history"]) > 1]
            if len(multi_hist) > 1:
                comp_path = output_dir / f"{problem.name}_comparison.png"
                plot_comparison(
                    [{"label": r["name"], "history": r["history"]} for r in multi_hist],
                    title=f"So sánh thuật toán trên {problem.name}",
                    save_path=str(comp_path),
                )
                print(f"  → {comp_path}")

            # Vẽ cho từng thuật toán
            coords = np.array(problem.coords)
            for r in results:
                # Convergence (chỉ vẽ nếu history > 1)
                if len(r["history"]) > 1:
                    conv_path = output_dir / f"{problem.name}_{r['name']}_convergence.png"
                    plot_convergence(
                        r["history"],
                        title=f"{r['name']} - {problem.name}",
                        save_path=str(conv_path),
                    )
                    print(f"  → {conv_path}")

                # Tour
                tour_path = output_dir / f"{problem.name}_{r['name']}_tour.png"
                plot_tour(
                    coords,
                    r["best_tour"],
                    title=f"{r['name']} - {problem.name} ({r['best_length']:.0f})",
                    save_path=str(tour_path),
                )
                print(f"  → {tour_path}")

        except ImportError as e:
            print(f"LỖI: Không thể import module visualize: {e}")

    print()
    print("=" * 70)
    print("HOÀN THÀNH")
    print("=" * 70)


if __name__ == "__main__":
    main()