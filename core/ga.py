"""
Module cài đặt Thuật giải Di truyền (GA) cho bài toán TSP.

Quy trình:
1. Khởi tạo quần thể ngẫu nhiên
2. Đánh giá fitness
3. Elitism: giữ lại cá thể tốt nhất
4. Chọn lọc + Lai ghép + Đột biến + 2-opt
5. Lặp qua các thế hệ
6. Trả về cá thể tốt nhất

Tác giả: [Tên bạn]
Ngày: [Ngày hôm nay]
"""

import sys
from pathlib import Path

# Thêm thư mục gốc dự án vào sys.path để import được package 'core'
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field

from core.operators import (
    tournament_selection,
    order_crossover,
    partially_mapped_crossover,
    swap_mutation,
    inversion_mutation,
    two_opt,
    tour_length,
)


@dataclass
class GAResult:
    """Kết quả sau khi chạy GA."""
    best_tour: np.ndarray
    best_length: float
    history: List[float] = field(default_factory=list)
    generations_run: int = 0
    time_seconds: float = 0.0


def init_population(n_cities: int, pop_size: int) -> List[np.ndarray]:
    """
    Khởi tạo quần thể ngẫu nhiên.

    Args:
        n_cities: Số thành phố
        pop_size: Kích thước quần thể

    Returns:
        Danh sách các hành trình
    """
    population = []
    for _ in range(pop_size):
        tour = np.random.permutation(n_cities)
        population.append(tour)
    return population


def evaluate_population(
    population: List[np.ndarray],
    dist: np.ndarray
) -> np.ndarray:
    """
    Đánh giá fitness của cả quần thể.

    Args:
        population: Danh sách hành trình
        dist: Ma trận khoảng cách

    Returns:
        Mảng fitness (giá trị càng nhỏ càng tốt)
    """
    return np.array([tour_length(t, dist) for t in population])


def run_ga(
    dist: np.ndarray,
    pop_size: int = 50,
    generations: int = 300,
    mutation_rate: float = 0.1,
    elite_size: int = 5,
    tournament_k: int = 3,
    crossover_type: str = "ox",
    mutation_type: str = "inversion",
    use_2opt: bool = True,
    two_opt_iterations: int = 5,
    two_opt_every: int = 5,
    two_opt_on_elite: bool = True,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> GAResult:
    """Chạy GA (đã tối ưu tốc độ)."""
    if seed is not None:
        np.random.seed(seed)

    n = dist.shape[0]
    start_time = time.time()

    if crossover_type == "ox":
        crossover_fn = order_crossover
    elif crossover_type == "pmx":
        crossover_fn = partially_mapped_crossover
    else:
        raise ValueError(f"crossover_type không hợp lệ: {crossover_type}")

    if mutation_type == "swap":
        mutation_fn = swap_mutation
    elif mutation_type == "inversion":
        mutation_fn = inversion_mutation
    else:
        raise ValueError(f"mutation_type không hợp lệ: {mutation_type}")

    # 1. Khởi tạo quần thể
    population = init_population(n, pop_size)
    fitness = evaluate_population(population, dist)

    best_idx = int(np.argmin(fitness))
    best_tour = population[best_idx].copy()
    best_length = float(fitness[best_idx])
    history = [best_length]

    if verbose:
        print(f"Thế hệ 0: best = {best_length:.2f}")

    # 2. Vòng lặp tiến hóa
    for gen in range(1, generations + 1):
        sorted_indices = np.argsort(fitness)
        population = [population[i] for i in sorted_indices]
        fitness = fitness[sorted_indices]

        new_population = [tour.copy() for tour in population[:elite_size]]

        # 2-opt cho elite (chỉ 1 lần duy nhất ở thế hệ đầu)
        if use_2opt and two_opt_on_elite and gen == 1:
            for i in range(len(new_population)):
                new_population[i] = two_opt(
                    new_population[i], dist, max_iterations=two_opt_iterations
                )

        # Xác định có chạy 2-opt ở thế hệ này không
        apply_2opt_this_gen = use_2opt and (gen % two_opt_every == 0)

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitness, k=tournament_k)
            parent2 = tournament_selection(population, fitness, k=tournament_k)
            child = crossover_fn(parent1, parent2)
            child = mutation_fn(child, mutation_rate=mutation_rate)

            if apply_2opt_this_gen:
                child = two_opt(child, dist, max_iterations=two_opt_iterations)

            new_population.append(child)

        population = new_population
        fitness = evaluate_population(population, dist)

        current_best_idx = int(np.argmin(fitness))
        current_best = float(fitness[current_best_idx])

        if current_best < best_length:
            best_length = current_best
            best_tour = population[current_best_idx].copy()

        history.append(best_length)

        if verbose and (gen % 50 == 0 or gen == generations):
            print(f"Thế hệ {gen}: best = {best_length:.2f}")

    elapsed = time.time() - start_time

    return GAResult(
        best_tour=best_tour,
        best_length=best_length,
        history=history,
        generations_run=generations,
        time_seconds=elapsed,
    )


if __name__ == "__main__":
    from core.data_loader import load_tsp
    from core.distance import compute_distance_matrix

    if len(sys.argv) < 2:
        print("Cách dùng: python core/ga.py <đường_dẫn_file.tsp>")
        sys.exit(1)

    # Đọc dữ liệu
    problem = load_tsp(sys.argv[1])
    dist = compute_distance_matrix(problem.coords, problem.edge_weight_type)

    print("=" * 60)
    print(f"CHẠY GA TRÊN {problem.name} ({problem.dimension} thành phố)")
    print("=" * 60)

    # Chạy GA
    result = run_ga(
        dist,
        pop_size=50,
        generations=1000,
        mutation_rate=0.1,
        elite_size=5,
        tournament_k=3,
        crossover_type="ox",
        mutation_type="inversion",
        use_2opt=True,
        two_opt_iterations=20,
        seed=42,
        verbose=True,
    )

    print()
    print("=" * 60)
    print("KẾT QUẢ CUỐI CÙNG")
    print("=" * 60)
    print(f"Độ dài tốt nhất:   {result.best_length:.2f}")
    print(f"Thời gian chạy:    {result.time_seconds:.2f} giây")
    print(f"Số thế hệ:         {result.generations_run}")
    print(f"Cải thiện:         {result.history[0] - result.best_length:.2f}")
    print(f"Tỉ lệ cải thiện:   {100 * (result.history[0] - result.best_length) / result.history[0]:.1f}%")