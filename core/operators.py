"""
Module chứa các toán tử của Thuật giải Di truyền (GA) cho TSP.

Bao gồm:
- Tournament Selection: chọn lọc cá thể
- Order Crossover (OX): lai ghép
- Swap Mutation: đột biến hoán đổi
- Inversion Mutation: đột biến đảo đoạn
- Two-opt: tối ưu cục bộ

Tác giả: [Tên bạn]
Ngày: [Ngày hôm nay]
"""

import numpy as np
from typing import List


# ============================================================
# CHỌN LỌC (SELECTION)
# ============================================================

def tournament_selection(
    population: List[np.ndarray],
    fitness: np.ndarray,
    k: int = 3
) -> np.ndarray:
    """
    Chọn lọc Tournament: chọn ngẫu nhiên k cá thể, lấy cá thể tốt nhất.

    Args:
        population: Danh sách các hành trình (mỗi hành trình là np.ndarray)
        fitness: Mảng fitness tương ứng (giá trị càng nhỏ càng tốt)
        k: Số cá thể tham gia tournament

    Returns:
        Một hành trình được chọn
    """
    n = len(population)
    # Chọn ngẫu nhiên k chỉ số (không trùng)
    indices = np.random.choice(n, size=k, replace=False)
    # Trong k cá thể đó, chọn cá thể có fitness nhỏ nhất
    best_idx = indices[np.argmin(fitness[indices])]
    return population[best_idx].copy()


# ============================================================
# LAI GHÉP (CROSSOVER)
# ============================================================

def order_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray
) -> np.ndarray:
    """
    Lai ghép Order Crossover (OX) cho TSP.

    Quy trình:
    1. Chọn 2 điểm cắt ngẫu nhiên
    2. Copy đoạn giữa của parent1 vào child
    3. Điền phần còn lại bằng thứ tự từ parent2 (bỏ qua thành phố đã có)

    Args:
        parent1: Hành trình cha
        parent2: Hành trình mẹ

    Returns:
        Hành trình con hợp lệ
    """
    n = len(parent1)
    child = np.full(n, -1, dtype=parent1.dtype)

    # Chọn 2 điểm cắt
    a, b = sorted(np.random.choice(n, size=2, replace=False))

    # Copy đoạn giữa từ parent1
    child[a:b + 1] = parent1[a:b + 1]

    # Xác định các thành phố đã có trong child
    used = set(parent1[a:b + 1].tolist())

    # Điền phần còn lại bằng thứ tự từ parent2, bắt đầu sau vị trí b
    fill_positions = list(range(b + 1, n)) + list(range(0, a))
    fill_values = [city for city in parent2 if city not in used]

    for pos, val in zip(fill_positions, fill_values):
        child[pos] = val

    return child


def partially_mapped_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray
) -> np.ndarray:
    """
    Lai ghép Partially Mapped Crossover (PMX).

    Tương tự OX nhưng dùng ánh xạ vị trí để điền phần còn lại.
    Thường cho kết quả tốt hơn OX trên một số bài toán.

    Args:
        parent1: Hành trình cha
        parent2: Hành trình mẹ

    Returns:
        Hành trình con hợp lệ
    """
    n = len(parent1)
    child = np.full(n, -1, dtype=parent1.dtype)

    a, b = sorted(np.random.choice(n, size=2, replace=False))

    # Copy đoạn giữa từ parent1
    child[a:b + 1] = parent1[a:b + 1]

    # Tạo ánh xạ giữa parent1 và parent2 trong đoạn giữa
    # Ví dụ: nếu parent1[a]=5, parent2[a]=7 thì 5 -> 7
    mapping = {}
    for i in range(a, b + 1):
        mapping[parent1[i]] = parent2[i]

    # Điền phần còn lại từ parent2
    for i in range(n):
        if a <= i <= b:
            continue
        candidate = parent2[i]
        # Theo ánh xạ cho đến khi tìm được thành phố chưa có
        while candidate in child:
            candidate = mapping[candidate]
        child[i] = candidate

    return child


# ============================================================
# ĐỘT BIẾN (MUTATION)
# ============================================================

def swap_mutation(
    tour: np.ndarray,
    mutation_rate: float = 0.1
) -> np.ndarray:
    """
    Đột biến Swap: đổi chỗ 2 thành phố ngẫu nhiên.

    Args:
        tour: Hành trình
        mutation_rate: Xác suất xảy ra đột biến

    Returns:
        Hành trình sau đột biến (bản sao)
    """
    tour = tour.copy()
    if np.random.random() < mutation_rate:
        i, j = np.random.choice(len(tour), size=2, replace=False)
        tour[i], tour[j] = tour[j], tour[i]
    return tour


def inversion_mutation(
    tour: np.ndarray,
    mutation_rate: float = 0.1
) -> np.ndarray:
    """
    Đột biến Inversion: đảo ngược một đoạn ngẫu nhiên.

    Thường cho kết quả tốt hơn Swap vì tương đương 2-opt.

    Args:
        tour: Hành trình
        mutation_rate: Xác suất xảy ra đột biến

    Returns:
        Hành trình sau đột biến (bản sao)
    """
    tour = tour.copy()
    if np.random.random() < mutation_rate:
        a, b = sorted(np.random.choice(len(tour), size=2, replace=False))
        tour[a:b + 1] = tour[a:b + 1][::-1]
    return tour


# ============================================================
# TỐI ƯU CỤC BỘ (LOCAL SEARCH)
# ============================================================

def two_opt(
    tour: np.ndarray,
    dist: np.ndarray,
    max_iterations: int = 20
) -> np.ndarray:
    """
    Tối ưu cục bộ bằng 2-opt (vectorized bằng numpy).

    Thay vì duyệt từng cặp (i, j), tính TẤT CẢ delta cùng lúc
    bằng numpy. Nhanh hơn 100-1000 lần so với vòng lặp Python.

    Args:
        tour: Hành trình ban đầu
        dist: Ma trận khoảng cách
        max_iterations: Số lần cải thiện tối đa

    Returns:
        Hành trình đã tối ưu
    """
    tour = tour.copy()
    n = len(tour)

    for _ in range(max_iterations):
        # tour_next[i] = thành phố tiếp theo sau tour[i]
        tour_next = np.roll(tour, -1)

        # a = tour[i], b = tour[i+1], c = tour[j], d = tour[j+1]
        # Dùng broadcasting để tạo ma trận (n, n)
        a = tour[:, np.newaxis]            # (n, 1)
        b = tour_next[:, np.newaxis]       # (n, 1)
        c = tour[np.newaxis, :]            # (1, n)
        d = tour_next[np.newaxis, :]       # (1, n)

        # delta[i][j] = dist[a][c] + dist[b][d] - dist[a][b] - dist[c][d]
        delta = dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]

        # Chỉ xét i < j - 1 (đảo ít nhất 2 thành phố)
        # Và bỏ trường hợp i=0, j=n-1 (đảo cả tour)
        mask = np.triu(np.ones((n, n), dtype=bool), k=2)
        mask[0, n - 1] = False

        # Đặt các vị trí không hợp lệ thành 0 (không cải thiện)
        delta_masked = np.where(mask, delta, 0.0)

        # Tìm delta nhỏ nhất (cải thiện nhiều nhất)
        min_val = delta_masked.min()

        if min_val >= -1e-6:
            # Không còn cải thiện → dừng
            break

        # Lấy vị trí (i, j) có delta nhỏ nhất
        i, j = np.unravel_index(delta_masked.argmin(), delta_masked.shape)

        # Đảo ngược đoạn từ i+1 đến j
        tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]

    return tour


# ============================================================
# TIỆN ÍCH
# ============================================================

def tour_length(tour: np.ndarray, dist: np.ndarray) -> float:
    """
    Tính tổng quãng đường của một hành trình.

    Dùng numpy roll để tính nhanh:
    dist[tour, roll(tour, -1)].sum()

    Args:
        tour: Hành trình
        dist: Ma trận khoảng cách

    Returns:
        Tổng quãng đường
    """
    return float(dist[tour, np.roll(tour, -1)].sum())


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.data_loader import load_tsp
    from core.distance import compute_distance_matrix

    if len(sys.argv) < 2:
        print("Cách dùng: python core/operators.py <đường_dẫn_file.tsp>")
        sys.exit(1)

    # Đọc dữ liệu
    problem = load_tsp(sys.argv[1])
    dist = compute_distance_matrix(problem.coords, problem.edge_weight_type)
    n = problem.dimension

    print("=" * 60)
    print(f"KIỂM TRA CÁC TOÁN TỬ GA TRÊN {problem.name}")
    print("=" * 60)

    # Tạo 2 hành trình ngẫu nhiên
    np.random.seed(42)
    p1 = np.random.permutation(n)
    p2 = np.random.permutation(n)

    print(f"\nCha (10 thành phố đầu):  {p1[:10]}")
    print(f"Mẹ  (10 thành phố đầu):  {p2[:10]}")

    # Test OX
    child_ox = order_crossover(p1, p2)
    valid = (
        len(set(child_ox.tolist())) == n and
        set(child_ox.tolist()) == set(range(n))
    )
    print(f"\n--- Order Crossover (OX) ---")
    print(f"Con (10 thành phố đầu):  {child_ox[:10]}")
    print(f"Hợp lệ? (đủ {n} thành phố, không trùng): {valid}")

    # Test PMX
    child_pmx = partially_mapped_crossover(p1, p2)
    valid = (
        len(set(child_pmx.tolist())) == n and
        set(child_pmx.tolist()) == set(range(n))
    )
    print(f"\n--- Partially Mapped Crossover (PMX) ---")
    print(f"Con (10 thành phố đầu):  {child_pmx[:10]}")
    print(f"Hợp lệ? {valid}")

     # Test Swap Mutation
    swapped = swap_mutation(p1, mutation_rate=1.0)
    n_diff = int(np.sum(p1 != swapped))
    print(f"\n--- Swap Mutation ---")
    print(f"Số vị trí khác nhau: {n_diff} (phải = 2)")
    print(f"Hợp lệ? {len(set(swapped.tolist())) == n}")

    # Test Inversion Mutation
    inverted = inversion_mutation(p1, mutation_rate=1.0)
    n_diff = int(np.sum(p1 != inverted))
    print(f"\n--- Inversion Mutation ---")
    print(f"Số vị trí khác nhau: {n_diff} (phải > 2)")
    print(f"Hợp lệ? {len(set(inverted.tolist())) == n}")

    # Test tour_length
    len1 = tour_length(p1, dist)
    len2 = tour_length(p2, dist)
    print(f"\n--- Độ dài hành trình ---")
    print(f"Cha: {len1:.2f}")
    print(f"Mẹ:  {len2:.2f}")

    # Test 2-opt
    print(f"\n--- 2-opt ---")
    print(f"Trước 2-opt: {len1:.2f}")
    optimized = two_opt(p1, dist, max_iterations=50)
    len_opt = tour_length(optimized, dist)
    print(f"Sau 2-opt:   {len_opt:.2f}")
    print(f"Cải thiện:   {len1 - len_opt:.2f} ({100*(len1-len_opt)/len1:.1f}%)")

    # Test Tournament
    print(f"\n--- Tournament Selection ---")
    population = [np.random.permutation(n) for _ in range(20)]
    fitness = np.array([tour_length(t, dist) for t in population])
    selected = tournament_selection(population, fitness, k=3)
    print(f"Fitness của quần thể: min={fitness.min():.0f}, max={fitness.max():.0f}")
    print(f"Fitness của cá thể được chọn: {tour_length(selected, dist):.0f}")

    print("\n" + "=" * 60)
    print("TẤT CẢ TOÁN TỬ HOẠT ĐỘNG")
    print("=" * 60)