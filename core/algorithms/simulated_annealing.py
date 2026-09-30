"""
Thuật toán Simulated Annealing (SA) cho bài toán TSP.

Phiên bản tối ưu: SA thuần + 2-opt định kỳ.

Ý tưởng:
- Bắt đầu với hành trình khởi tạo (mặc định: NN).
- Mỗi vòng, tạo hành trình lân cận bằng đảo đoạn ngẫu nhiên.
- Chấp nhận nếu tốt hơn, hoặc chấp nhận với xác suất exp(-Δ/T) nếu kém hơn.
- Cứ mỗi N vòng, chạy 2-opt để "dọn dẹp" hành trình hiện tại.
- Nhiệt độ T giảm dần theo cooling schedule.

Ưu điểm:
- Giữ đúng bản chất SA (có từ chối).
- 2-opt định kỳ giúp tăng chất lượng mà không làm mất đặc tính SA.
- Nhanh hơn hybrid SA+2opt mỗi bước.

Tác giả: [Tên bạn]
Ngày: [Ngày hôm nay]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
import numpy as np
from typing import Optional, List
from dataclasses import dataclass, field

from core.operators import two_opt


@dataclass
class SAResult:
    """Kết quả của Simulated Annealing."""
    best_tour: np.ndarray
    best_length: float
    history: List[float] = field(default_factory=list)
    iterations_run: int = 0
    time_seconds: float = 0.0
    accepted_moves: int = 0
    rejected_moves: int = 0
    improvements: int = 0


def generate_neighbor(tour: np.ndarray, move_type: str = "inversion") -> np.ndarray:
    """
    Tạo hành trình lân cận bằng đột biến nhẹ.

    Args:
        tour: Hành trình hiện tại
        move_type: "inversion" (đảo đoạn), "swap" (đổi chỗ), "insertion" (chèn)

    Returns:
        Hành trình mới
    """
    neighbor = tour.copy()
    n = len(neighbor)

    if move_type == "swap":
        i, j = np.random.choice(n, size=2, replace=False)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

    elif move_type == "inversion":
        a, b = sorted(np.random.choice(n, size=2, replace=False))
        neighbor[a:b + 1] = neighbor[a:b + 1][::-1]

    elif move_type == "insertion":
        i, j = np.random.choice(n, size=2, replace=False)
        city = neighbor[i]
        neighbor = np.delete(neighbor, i)
        neighbor = np.insert(neighbor, j, city)

    else:
        raise ValueError(f"move_type không hợp lệ: {move_type}")

    return neighbor


def simulated_annealing(
    dist: np.ndarray,
    initial_tour: Optional[np.ndarray] = None,
    max_iterations: int = 100_000,
    initial_temp: float = 1000.0,
    final_temp: float = 0.01,
    cooling_rate: float = 0.9999,
    move_type: str = "inversion",
    local_search_every: int = 1000,
    local_search_iter: int = 50,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> SAResult:
    """
    Chạy Simulated Annealing với 2-opt định kỳ.

    Args:
        dist: Ma trận khoảng cách (n, n)
        initial_tour: Hành trình ban đầu. Nếu None, tạo ngẫu nhiên.
        max_iterations: Số vòng lặp tối đa.
        initial_temp: Nhiệt độ ban đầu.
        final_temp: Nhiệt độ dừng.
        cooling_rate: Hệ số giảm nhiệt mỗi vòng.
        move_type: "inversion", "swap", hoặc "insertion".
        local_search_every: Chạy 2-opt mỗi N vòng.
        local_search_iter: Số vòng 2-opt mỗi lần chạy.
        seed: Hạt giống ngẫu nhiên.
        verbose: In tiến độ.

    Returns:
        SAResult chứa hành trình tốt nhất, lịch sử, thống kê.
    """
    if seed is not None:
        np.random.seed(seed)

    n = dist.shape[0]
    start_time = time.time()

    # Khởi tạo
    if initial_tour is None:
        current_tour = np.random.permutation(n).astype(np.int32)
    else:
        current_tour = initial_tour.copy().astype(np.int32)

    def tour_length(tour):
        return float(dist[tour, np.roll(tour, -1)].sum())

    # Áp dụng 2-opt ban đầu để có xuất phát tốt
    current_tour = two_opt(current_tour, dist, max_iterations=local_search_iter)
    current_length = tour_length(current_tour)

    best_tour = current_tour.copy()
    best_length = current_length
    history = [best_length]

    temp = initial_temp
    accepted = 0
    rejected = 0
    improvements = 0

    if verbose:
        print(f"Khởi tạo: length = {current_length:.2f}, T = {temp:.2f}")

    for iteration in range(max_iterations):
        # 1. Tạo neighbor
        neighbor = generate_neighbor(current_tour, move_type)
        neighbor_length = tour_length(neighbor)
        delta = neighbor_length - current_length

        # 2. Quyết định chấp nhận theo SA
        if delta < 0:
            accept = True
        elif temp > 1e-10:
            prob = np.exp(-delta / temp)
            accept = np.random.random() < prob
        else:
            accept = False

        if accept:
            current_tour = neighbor
            current_length = neighbor_length
            accepted += 1

            if current_length < best_length:
                best_length = current_length
                best_tour = current_tour.copy()
                improvements += 1
        else:
            rejected += 1

        # 3. 2-opt định kỳ để dọn dẹp
        if (iteration + 1) % local_search_every == 0:
            optimized = two_opt(current_tour, dist, max_iterations=local_search_iter)
            opt_length = tour_length(optimized)

            if opt_length < current_length:
                current_tour = optimized
                current_length = opt_length

                if current_length < best_length:
                    best_length = current_length
                    best_tour = current_tour.copy()
                    improvements += 1

        history.append(best_length)
        temp *= cooling_rate

        if temp < final_temp:
            if verbose:
                print(f"Dừng ở vòng {iteration + 1}: T = {temp:.6f}")
            break

        if verbose and (iteration + 1) % 20000 == 0:
            rate = 100 * accepted / (accepted + rejected)
            print(f"Vòng {iteration + 1}: best = {best_length:.2f}, "
                  f"T = {temp:.4f}, accept_rate = {rate:.1f}%")

    elapsed = time.time() - start_time
    final_iter = iteration + 1

    return SAResult(
        best_tour=best_tour,
        best_length=best_length,
        history=history,
        iterations_run=final_iter,
        time_seconds=elapsed,
        accepted_moves=accepted,
        rejected_moves=rejected,
        improvements=improvements,
    )


if __name__ == "__main__":
    from core.data_loader import load_tsp
    from core.distance import compute_distance_matrix
    from core.algorithms.nearest_neighbor import nearest_neighbor

    if len(sys.argv) < 2:
        print("Cách dùng: python core/algorithms/simulated_annealing.py <file.tsp>")
        sys.exit(1)

    # Đọc dữ liệu
    problem = load_tsp(sys.argv[1])
    dist = compute_distance_matrix(problem.coords, problem.edge_weight_type)

    print("=" * 60)
    print(f"SIMULATED ANNEALING TRÊN {problem.name} ({problem.dimension} thành phố)")
    print("=" * 60)

    # Khởi tạo bằng NN
    print("\n--- Khởi tạo bằng Nearest Neighbor ---")
    nn_result = nearest_neighbor(dist, n_starts=None, verbose=False)
    print(f"NN best: {nn_result.best_length:.2f}")

    # Chạy SA
    print("\n--- Chạy Simulated Annealing ---")
    result = simulated_annealing(
        dist,
        initial_tour=nn_result.best_tour,
        max_iterations=100_000,
        initial_temp=1000.0,
        final_temp=0.01,
        cooling_rate=0.9999,
        move_type="inversion",
        local_search_every=1000,
        local_search_iter=50,
        seed=42,
        verbose=True,
    )

    print()
    print("=" * 60)
    print("KẾT QUẢ")
    print("=" * 60)
    print(f"Best length:      {result.best_length:.2f}")
    print(f"Thời gian:        {result.time_seconds:.2f} giây")
    print(f"Số vòng lặp:      {result.iterations_run}")
    print(f"Chấp nhận:        {result.accepted_moves}")
    print(f"Từ chối:          {result.rejected_moves}")

    total = result.accepted_moves + result.rejected_moves
    if total > 0:
        rate = 100 * result.accepted_moves / total
        print(f"Tỉ lệ chấp nhận:  {rate:.1f}%")
    print(f"Số lần cải thiện: {result.improvements}")

    # So sánh với đáp án tối ưu
    KNOWN_OPTIMA = {
        "berlin52": 7542, "eil51": 426, "ch150": 6528,
        "kroA100": 21282, "tsp225": 3919, "pr439": 107217,
    }
    optimal = KNOWN_OPTIMA.get(problem.name)

    if optimal:
        gap = 100 * (result.best_length - optimal) / optimal
        print()
        print(f"Đáp án tối ưu:    {optimal}")
        print(f"Sai số:           {gap:.2f}%")