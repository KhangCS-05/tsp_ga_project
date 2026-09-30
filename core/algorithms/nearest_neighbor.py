"""
Thuật toán Nearest Neighbor (NN) cho bài toán TSP.

Ý tưởng:
- Bắt đầu từ một thành phố xuất phát.
- Mỗi bước, đi đến thành phố gần nhất chưa thăm.
- Khi đã thăm hết, quay về điểm xuất phát.

Đặc điểm:
- Độ phức tạp: O(n²).
- Rất nhanh, nhưng chất lượng lời giải kém (thường sai 20-25% so với tối ưu).
- Dùng làm baseline để so sánh với GA, SA.

Cải tiến:
- Chạy NN từ nhiều điểm xuất phát khác nhau.
- Lấy kết quả tốt nhất → giảm sai số xuống còn ~10-15%.

Tác giả: [Tên bạn]
Ngày: [Ngày hôm nay]
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class NNResult:
    """Kết quả của Nearest Neighbor."""
    best_tour: np.ndarray
    best_length: float
    start_city: int
    all_lengths: list  # độ dài của tất cả các lần chạy


def nearest_neighbor_from(
    dist: np.ndarray,
    start_city: int
) -> tuple:
    """
    Chạy Nearest Neighbor từ MỘT thành phố xuất phát.

    Args:
        dist: Ma trận khoảng cách (n, n)
        start_city: Thành phố xuất phát (0 <= start_city < n)

    Returns:
        (tour, length): Hành trình và tổng độ dài
    """
    n = dist.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = np.empty(n, dtype=np.int32)

    current = start_city
    tour[0] = current
    visited[current] = True
    total = 0.0

    for i in range(1, n):
        # Lấy khoảng cách từ current đến tất cả thành phố
        distances = dist[current].copy()

        # Đặt thành phố đã thăm thành vô cực (để không chọn)
        distances[visited] = np.inf

        # Chọn thành phố gần nhất
        next_city = int(np.argmin(distances))

        tour[i] = next_city
        visited[next_city] = True
        total += distances[next_city]

        current = next_city

    # Quay về điểm xuất phát
    total += dist[current, start_city]

    return tour, total


def nearest_neighbor(
    dist: np.ndarray,
    n_starts: Optional[int] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> NNResult:
    """
    Chạy Nearest Neighbor từ NHIỀU điểm xuất phát, lấy kết quả tốt nhất.

    Args:
        dist: Ma trận khoảng cách (n, n)
        n_starts: Số điểm xuất phát thử. Nếu None, thử TẤT CẢ n thành phố.
        seed: Hạt giống ngẫu nhiên (chỉ dùng khi n_starts < n).
        verbose: In tiến độ.

    Returns:
        NNResult chứa hành trình tốt nhất, độ dài, và tất cả độ dài.
    """
    n = dist.shape[0]

    if seed is not None:
        np.random.seed(seed)

    # Xác định danh sách điểm xuất phát
    if n_starts is None or n_starts >= n:
        start_cities = list(range(n))
    else:
        start_cities = np.random.choice(n, size=n_starts, replace=False).tolist()

    best_tour = None
    best_length = float("inf")
    all_lengths = []

    for idx, start in enumerate(start_cities):
        tour, length = nearest_neighbor_from(dist, start)
        all_lengths.append(length)

        if length < best_length:
            best_length = length
            best_tour = tour.copy()

        if verbose and (idx + 1) % 10 == 0:
            print(f"  Đã thử {idx + 1}/{len(start_cities)} điểm xuất phát, "
                  f"best = {best_length:.2f}")

    return NNResult(
        best_tour=best_tour,
        best_length=best_length,
        start_city=int(best_tour[0]),
        all_lengths=all_lengths,
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path
    import time

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from core.data_loader import load_tsp
    from core.distance import compute_distance_matrix
    from core.operators import tour_length

    if len(sys.argv) < 2:
        print("Cách dùng: python core/algorithms/nearest_neighbor.py <file.tsp>")
        sys.exit(1)

    # Đọc dữ liệu
    problem = load_tsp(sys.argv[1])
    dist = compute_distance_matrix(problem.coords, problem.edge_weight_type)

    print("=" * 60)
    print(f"NEAREST NEIGHBOR TRÊN {problem.name} ({problem.dimension} thành phố)")
    print("=" * 60)

    # Chạy từ 1 điểm
    print("\n--- Chạy từ 1 điểm xuất phát (thành phố 0) ---")
    t0 = time.time()
    tour1, len1 = nearest_neighbor_from(dist, 0)
    t1 = time.time() - t0
    print(f"Độ dài:    {len1:.2f}")
    print(f"Thời gian: {t1 * 1000:.2f} ms")

    # Chạy từ nhiều điểm
    print("\n--- Chạy từ tất cả điểm xuất phát ---")
    t0 = time.time()
    result = nearest_neighbor(dist, n_starts=None, verbose=True)
    t2 = time.time() - t0
    print(f"Best:      {result.best_length:.2f}")
    print(f"Xuất phát: thành phố {result.start_city}")
    print(f"Thời gian: {t2:.3f} giây")

    # So sánh với đáp án tối ưu
    KNOWN_OPTIMA = {
        "berlin52": 7542, "eil51": 426, "ch150": 6528,
        "kroA100": 21282, "tsp225": 3919, "pr439": 107217,
    }
    optimal = KNOWN_OPTIMA.get(problem.name)

    print()
    print("=" * 60)
    print("SO SÁNH VỚI ĐÁP ÁN TỐI ƯU")
    print("=" * 60)
    if optimal:
        gap1 = 100 * (len1 - optimal) / optimal
        gap2 = 100 * (result.best_length - optimal) / optimal
        print(f"Tối ưu:                  {optimal}")
        print(f"NN 1 điểm:               {len1:.2f} (sai {gap1:.2f}%)")
        print(f"NN nhiều điểm:           {result.best_length:.2f} (sai {gap2:.2f}%)")
    else:
        print(f"NN 1 điểm:      {len1:.2f}")
        print(f"NN nhiều điểm:  {result.best_length:.2f}")
        print(f"(Không có đáp án tối ưu để so sánh)")