"""
File chạy chính của dự án GA-TSP.

Cách dùng:
    python main.py <file.tsp> [tùy chọn]

Ví dụ:
    python main.py data/ch150.tsp
    python main.py data/ch150.tsp --pop-size 100 --generations 1000
    python main.py data/ch150.tsp --plot --save outputs/result.json

Tác giả: [Tên bạn]
Ngày: [Ngày hôm nay]
"""

import sys
import argparse
import json
import time
from pathlib import Path

# Thêm thư mục gốc vào sys.path
sys.path.insert(0, str(Path(__file__).parent))

from core.data_loader import load_tsp, print_problem_info
from core.distance import compute_distance_matrix
from core.ga import run_ga, GAResult


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
        description="Ứng dụng Thuật giải Di truyền giải bài toán TSP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Tham số bắt buộc
    parser.add_argument(
        "tsp_file",
        type=str,
        help="Đường dẫn tới file .tsp (định dạng TSPLIB)",
    )

    # Tham số GA
    parser.add_argument(
        "--pop-size", type=int, default=100,
        help="Kích thước quần thể",
    )
    parser.add_argument(
        "--generations", type=int, default=500,
        help="Số thế hệ",
    )
    parser.add_argument(
        "--mutation-rate", type=float, default=0.1,
        help="Tỉ lệ đột biến (0.0 - 1.0)",
    )
    parser.add_argument(
        "--elite-size", type=int, default=5,
        help="Số cá thể elitism giữ lại mỗi thế hệ",
    )
    parser.add_argument(
        "--tournament-k", type=int, default=3,
        help="Số cá thể tham gia tournament selection",
    )
    parser.add_argument(
        "--crossover", type=str, default="ox", choices=["ox", "pmx"],
        help="Loại lai ghép: ox (Order) hoặc pmx (Partially Mapped)",
    )
    parser.add_argument(
        "--mutation", type=str, default="inversion",
        choices=["swap", "inversion"],
        help="Loại đột biến: swap hoặc inversion",
    )
    parser.add_argument(
        "--no-2opt", action="store_true",
        help="Tắt 2-opt (chạy GA thuần, không tối ưu cục bộ)",
    )
    parser.add_argument(
        "--two-opt-iter", type=int, default=20,
        help="Số vòng lặp 2-opt mỗi lần áp dụng",
    )
    parser.add_argument(
        "--two-opt-every", type=int, default=1,
        help="Áp dụng 2-opt mỗi N thế hệ",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Hạt giống ngẫu nhiên (để tái lập kết quả)",
    )

    # Tham số output
    parser.add_argument(
        "--plot", action="store_true",
        help="Vẽ biểu đồ hội tụ và hành trình",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Lưu kết quả ra file JSON",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Thư mục lưu ảnh và kết quả",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Không in tiến độ GA",
    )

    return parser.parse_args()


def save_result(
    result: GAResult,
    problem_name: str,
    filepath: str,
    args: argparse.Namespace,
) -> None:
    """Lưu kết quả ra file JSON."""
    output = {
        "problem": problem_name,
        "tsp_file": str(args.tsp_file),
        "best_length": float(result.best_length),
        "time_seconds": float(result.time_seconds),
        "generations_run": result.generations_run,
        "best_tour": result.best_tour.tolist(),
        "history": result.history,
        "params": {
            "pop_size": args.pop_size,
            "generations": args.generations,
            "mutation_rate": args.mutation_rate,
            "elite_size": args.elite_size,
            "tournament_k": args.tournament_k,
            "crossover": args.crossover,
            "mutation": args.mutation,
            "use_2opt": not args.no_2opt,
            "two_opt_iter": args.two_opt_iter,
            "two_opt_every": args.two_opt_every,
            "seed": args.seed,
        },
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def main():
    """Hàm chính."""
    args = parse_args()

    print("=" * 70)
    print("  ỨNG DỤNG THUẬT GIẢI DI TRUYỀN GIẢI BÀI TOÁN NGƯỜI DU LỊCH")
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

    # 3. In tham số GA
    print("Tham số GA:")
    print(f"  Kích thước quần thể:   {args.pop_size}")
    print(f"  Số thế hệ:             {args.generations}")
    print(f"  Tỉ lệ đột biến:        {args.mutation_rate}")
    print(f"  Elitism:               {args.elite_size}")
    print(f"  Tournament k:          {args.tournament_k}")
    print(f"  Lai ghép:              {args.crossover.upper()}")
    print(f"  Đột biến:              {args.mutation}")
    print(f"  2-opt:                 {'TẮT' if args.no_2opt else f'BẬT ({args.two_opt_iter} vòng, mỗi {args.two_opt_every} thế hệ)'}")
    print(f"  Seed:                  {args.seed if args.seed is not None else 'ngẫu nhiên'}")
    print()

    # 4. Chạy GA
    print("=" * 70)
    print("BẮT ĐẦU CHẠY GA")
    print("=" * 70)

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
        verbose=not args.quiet,
    )

    # 5. In kết quả
    print()
    print("=" * 70)
    print("KẾT QUẢ")
    print("=" * 70)
    print(f"Độ dài tốt nhất:    {result.best_length:.2f}")
    print(f"Thời gian chạy:     {result.time_seconds:.2f} giây")
    print(f"Số thế hệ:          {result.generations_run}")

    # So sánh với đáp án tối ưu
    optimal = KNOWN_OPTIMA.get(problem.name)
    if optimal is not None:
        gap = result.best_length - optimal
        gap_pct = 100 * gap / optimal
        print()
        print(f"Đáp án tối ưu:      {optimal}")
        print(f"Chênh lệch:         {gap:.2f} ({gap_pct:.3f}%)")
        if gap_pct < 0.5:
            print("Đánh giá:           XUẤT SẮC (sai số < 0.5%)")
        elif gap_pct < 2:
            print("Đánh giá:           TỐT (sai số < 2%)")
        elif gap_pct < 5:
            print("Đánh giá:           CHẤP NHẬN ĐƯỢC (sai số < 5%)")
        else:
            print("Đánh giá:           CẦN CẢI THIỆN")
    print()

    # 6. Lưu kết quả JSON
    if args.save:
        save_result(result, problem.name, args.save, args)
        print(f"Đã lưu kết quả JSON: {args.save}")

    # 7. Vẽ biểu đồ
    if args.plot:
        try:
            from core.visualize import (
                plot_convergence,
                plot_tour,
                plot_history_with_optimal,
            )
            import numpy as np

            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)

            print("\nĐang vẽ biểu đồ...")

            # Biểu đồ hội tụ
            conv_path = output_dir / f"{problem.name}_convergence.png"
            plot_convergence(
                result.history,
                title=f"Sự hội tụ của GA trên {problem.name}",
                save_path=str(conv_path),
            )
            print(f"  → {conv_path}")

            # Biểu đồ hội tụ với đáp án tối ưu
            if optimal is not None:
                opt_path = output_dir / f"{problem.name}_optimal.png"
                plot_history_with_optimal(
                    result.history,
                    optimal,
                    title=f"GA trên {problem.name} vs Tối ưu {optimal}",
                    save_path=str(opt_path),
                )
                print(f"  → {opt_path}")

            # Biểu đồ hành trình
            tour_path = output_dir / f"{problem.name}_tour.png"
            coords = np.array(problem.coords)
            plot_tour(
                coords,
                result.best_tour,
                title=f"Hành trình tốt nhất - {problem.name} ({result.best_length:.0f})",
                save_path=str(tour_path),
            )
            print(f"  → {tour_path}")

        except ImportError as e:
            print(f"LỖI: Không thể import module visualize: {e}")
            print("Cài đặt matplotlib: pip install matplotlib")

    print()
    print("=" * 70)
    print("HOÀN THÀNH")
    print("=" * 70)


if __name__ == "__main__":
    main()