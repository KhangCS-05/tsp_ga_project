"""
Module trực quan hóa kết quả GA cho TSP.

Bao gồm:
- Vẽ biểu đồ hội tụ (convergence plot)
- Vẽ hành trình tốt nhất (tour plot)
- Vẽ so sánh nhiều lần chạy (comparison plot)

Tác giả: [Tên bạn]
Ngày: [Ngày hôm nay]
"""

import matplotlib
matplotlib.use("Agg")  # Dùng backend không cần GUI, tránh lỗi trên Windows

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
from pathlib import Path


def plot_convergence(
    history: List[float],
    title: str = "Sự hội tụ của GA",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Vẽ biểu đồ hội tụ: best length qua các thế hệ.

    Args:
        history: Danh sách best length theo từng thế hệ
        title: Tiêu đề biểu đồ
        save_path: Nếu có, lưu ảnh vào đường dẫn này
        show: Nếu True, hiển thị biểu đồ (cần GUI)

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    generations = range(len(history))
    ax.plot(generations, history, color="steelblue", linewidth=2)

    # Đánh dấu điểm cuối
    ax.scatter(
        [len(history) - 1], [history[-1]],
        color="red", s=100, zorder=5,
        label=f"Best = {history[-1]:.0f}"
    )

    ax.set_xlabel("Thế hệ", fontsize=12)
    ax.set_ylabel("Độ dài hành trình tốt nhất", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_tour(
    coords: np.ndarray,
    tour: np.ndarray,
    title: str = "Hành trình tốt nhất",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Vẽ hành trình tốt nhất trên mặt phẳng 2D.

    Args:
        coords: Mảng tọa độ (n, 2)
        tour: Hành trình (mảng chỉ số thành phố)
        title: Tiêu đề
        save_path: Nếu có, lưu ảnh
        show: Nếu True, hiển thị

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    coords = np.asarray(coords)
    tour = np.asarray(tour)

    # Lấy tọa độ theo thứ tự tour, đóng vòng
    tour_coords = coords[tour]
    tour_coords_closed = np.vstack([tour_coords, tour_coords[0]])

    # Vẽ đường nối
    ax.plot(
        tour_coords_closed[:, 0],
        tour_coords_closed[:, 1],
        color="steelblue",
        linewidth=1.2,
        alpha=0.7,
        zorder=2,
    )

    # Vẽ các thành phố
    ax.scatter(
        coords[:, 0], coords[:, 1],
        color="red", s=25, zorder=3,
        edgecolors="darkred", linewidths=0.5,
    )

    # Đánh dấu thành phố xuất phát
    ax.scatter(
        [coords[tour[0], 0]], [coords[tour[0], 1]],
        color="lime", s=200, marker="*",
        zorder=4, edgecolors="black", linewidths=1,
        label="Điểm xuất phát",
    )

    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_comparison(
    results: List[dict],
    title: str = "So sánh các cấu hình GA",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Vẽ so sánh nhiều lần chạy GA.

    Args:
        results: Danh sách dict, mỗi dict có:
            - 'label': tên cấu hình
            - 'history': danh sách best length
        title: Tiêu đề
        save_path: Nếu có, lưu ảnh
        show: Nếu True, hiển thị

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for r, color in zip(results, colors):
        ax.plot(
            range(len(r["history"])),
            r["history"],
            label=f"{r['label']} (best={r['history'][-1]:.0f})",
            color=color,
            linewidth=2,
        )

    ax.set_xlabel("Thế hệ", fontsize=12)
    ax.set_ylabel("Độ dài hành trình tốt nhất", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_history_with_optimal(
    history: List[float],
    optimal: float,
    title: str = "Sự hội tụ của GA so với đáp án tối ưu",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Vẽ biểu đồ hội tụ có đường tham chiếu đáp án tối ưu.

    Args:
        history: Danh sách best length
        optimal: Giá trị tối ưu đã biết
        title: Tiêu đề
        save_path: Nếu có, lưu ảnh
        show: Nếu True, hiển thị

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        range(len(history)), history,
        color="steelblue", linewidth=2,
        label="GA",
    )

    ax.axhline(
        y=optimal, color="red", linestyle="--", linewidth=2,
        label=f"Tối ưu = {optimal}",
    )

    # Vùng sai số
    ax.fill_between(
        range(len(history)),
        optimal,
        history,
        color="orange", alpha=0.15,
        label="Khoảng cách tới tối ưu",
    )

    ax.set_xlabel("Thế hệ", fontsize=12)
    ax.set_ylabel("Độ dài hành trình tốt nhất", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from core.data_loader import load_tsp
    from core.distance import compute_distance_matrix
    from core.ga import run_ga

    if len(sys.argv) < 2:
        print("Cách dùng: python core/visualize.py <đường_dẫn_file.tsp>")
        sys.exit(1)

    # Tạo thư mục outputs nếu chưa có
    outputs_dir = Path(__file__).parent.parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    # Đọc dữ liệu
    problem = load_tsp(sys.argv[1])
    dist = compute_distance_matrix(problem.coords, problem.edge_weight_type)

    print(f"Chạy GA trên {problem.name}...")

    # Chạy GA
    result = run_ga(
        dist,
        pop_size=50,
        generations=500,
        mutation_rate=0.1,
        elite_size=5,
        tournament_k=3,
        crossover_type="ox",
        mutation_type="inversion",
        use_2opt=True,
        two_opt_iterations=20,
        two_opt_every=1,
        seed=42,
        verbose=False,
    )

    print(f"Best length: {result.best_length:.2f}")
    print(f"Thời gian:    {result.time_seconds:.2f} giây")

    # Đáp án tối ưu của một số bài toán (tra trong TSPLIB)
    KNOWN_OPTIMA = {
        "ch150": 6528,
        "berlin52": 7542,
        "eil51": 426,
        "kroA100": 21282,
        "tsp225": 3919,
        "pr439": 107217,
    }
    optimal = KNOWN_OPTIMA.get(problem.name)

    # Vẽ biểu đồ hội tụ
    print("\nĐang vẽ biểu đồ hội tụ...")
    plot_convergence(
        result.history,
        title=f"Sự hội tụ của GA trên {problem.name}",
        save_path=str(outputs_dir / f"{problem.name}_convergence.png"),
    )
    print(f"  → Đã lưu: outputs/{problem.name}_convergence.png")

    # Vẽ biểu đồ hội tụ với đáp án tối ưu
    if optimal is not None:
        print("\nĐang vẽ biểu đồ hội tụ với đáp án tối ưu...")
        plot_history_with_optimal(
            result.history,
            optimal,
            title=f"GA trên {problem.name} vs Tối ưu {optimal}",
            save_path=str(outputs_dir / f"{problem.name}_convergence_optimal.png"),
        )
        print(f"  → Đã lưu: outputs/{problem.name}_convergence_optimal.png")

    # Vẽ hành trình tốt nhất
    print("\nĐang vẽ hành trình tốt nhất...")
    coords = np.array(problem.coords)
    plot_tour(
        coords,
        result.best_tour,
        title=f"Hành trình tốt nhất - {problem.name} ({result.best_length:.0f})",
        save_path=str(outputs_dir / f"{problem.name}_tour.png"),
    )
    print(f"  → Đã lưu: outputs/{problem.name}_tour.png")

    print("\n" + "=" * 60)
    print("HOÀN THÀNH. Kiểm tra thư mục 'outputs/' để xem ảnh.")
    print("=" * 60)