"""
Module tính ma trận khoảng cách cho bài toán TSP.

Hỗ trợ các loại khoảng cách TSPLIB:
- EUC_2D:  Euclidean 2D, làm tròn về số nguyên gần nhất
- CEIL_2D: Euclidean 2D, làm tròn lên
- GEO:     Khoảng cách địa lý (dùng cho tọa độ latitude/longitude)
- ATT:     Khoảng cách giả Euclidean (dùng cho att48, att532)

Tác giả: [Tên bạn]
Ngày: [Ngày hôm nay]
"""

import numpy as np
from typing import Tuple


def _euclidean_2d(coords: np.ndarray) -> np.ndarray:
    """
    Tính ma trận khoảng cách Euclidean 2D, làm tròn về số nguyên gần nhất.

    Công thức TSPLIB: d = round(sqrt(dx^2 + dy^2))

    Args:
        coords: Mảng numpy shape (n, 2)

    Returns:
        Ma trận khoảng cách shape (n, n), dtype float32
    """
    # Tách tọa độ x và y
    x = coords[:, 0]
    y = coords[:, 1]

    # Tính hiệu tọa độ theo từng cặp (broadcasting)
    # dx[i][j] = x[i] - x[j]
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]

    # Khoảng cách Euclidean
    dist = np.sqrt(dx ** 2 + dy ** 2)

    # Làm tròn về số nguyên gần nhất (theo chuẩn TSPLIB)
    dist = np.round(dist)

    return dist.astype(np.float32)


def _ceil_2d(coords: np.ndarray) -> np.ndarray:
    """Tính ma trận khoảng cách Euclidean 2D, làm tròn lên."""
    x = coords[:, 0]
    y = coords[:, 1]
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist = np.ceil(np.sqrt(dx ** 2 + dy ** 2))
    return dist.astype(np.float32)


def _geo(coords: np.ndarray) -> np.ndarray:
    """
    Tính ma trận khoảng cách địa lý theo công thức TSPLIB.

    Tọa độ đầu vào là (latitude, longitude) theo độ.
    Kết quả là khoảng cách theo km (đã làm tròn).
    """
    # Chuyển độ sang radian
    deg = np.pi / 180.0

    # Bán kính trái đất theo TSPLIB
    RRR = 6378.388

    lat = coords[:, 0]
    lon = coords[:, 1]

    # Công thức TSPLIB cho GEO
    q1 = np.cos(lon[:, np.newaxis] - lon[np.newaxis, :] * 0 + lon[np.newaxis, :] * 0)
    # (giữ đơn giản: tính từng cặp)

    n = coords.shape[0]
    dist = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            lat_i = deg * lat[i]
            lon_i = deg * lon[i]
            lat_j = deg * lat[j]
            lon_j = deg * lon[j]

            q1 = np.cos(lon_i - lon_j)
            q2 = np.cos(lat_i - lat_j)
            q3 = np.cos(lat_i + lat_j)

            dij = RRR * np.arccos(
                0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)
            ) + 1.0

            dist[i][j] = int(dij + 0.5)

    return dist


def _att(coords: np.ndarray) -> np.ndarray:
    """
    Tính ma trận khoảng cách ATT theo công thức TSPLIB.

    Công thức:
        rij = sqrt((dx^2 + dy^2) / 10.0)
        tij = round(rij)
        nếu tij < rij: tij += 1
    """
    x = coords[:, 0]
    y = coords[:, 1]
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]

    rij = np.sqrt((dx ** 2 + dy ** 2) / 10.0)
    tij = np.round(rij)

    # Nếu tij < rij, cộng thêm 1
    mask = tij < rij
    tij[mask] += 1

    return tij.astype(np.float32)


def compute_distance_matrix(
    coords: list,
    edge_weight_type: str = "EUC_2D"
) -> np.ndarray:
    """
    Tính ma trận khoảng cách từ danh sách tọa độ.

    Args:
        coords: Danh sách tọa độ [(x1, y1), (x2, y2), ...]
        edge_weight_type: Loại khoảng cách ('EUC_2D', 'CEIL_2D', 'GEO', 'ATT')

    Returns:
        Ma trận khoảng cách numpy shape (n, n), dtype float32

    Raises:
        ValueError: Nếu edge_weight_type không được hỗ trợ
    """
    coords_array = np.array(coords, dtype=np.float64)

    if edge_weight_type == "EUC_2D":
        return _euclidean_2d(coords_array)
    elif edge_weight_type == "CEIL_2D":
        return _ceil_2d(coords_array)
    elif edge_weight_type == "GEO":
        return _geo(coords_array)
    elif edge_weight_type == "ATT":
        return _att(coords_array)
    else:
        raise ValueError(
            f"Loại khoảng cách '{edge_weight_type}' chưa được hỗ trợ. "
            f"Các loại hỗ trợ: EUC_2D, CEIL_2D, GEO, ATT"
        )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Thêm thư mục gốc vào sys.path để import core.data_loader
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from core.data_loader import load_tsp

    if len(sys.argv) < 2:
        print("Cách dùng: python core/distance.py <đường_dẫn_file.tsp>")
        sys.exit(1)

    filepath = sys.argv[1]
    problem = load_tsp(filepath)

    print(f"Đang tính ma trận khoảng cách cho {problem.name}...")
    dist = compute_distance_matrix(problem.coords, problem.edge_weight_type)

    print(f"Kích thước ma trận: {dist.shape}")
    print(f"Kiểu dữ liệu:       {dist.dtype}")
    print(f"Bộ nhớ:             {dist.nbytes / 1024:.2f} KB")
    print(f"Đường chéo (phải = 0): {dist.diagonal()[:5]}")
    print(f"Đối xứng?           {np.allclose(dist, dist.T)}")
    print()
    print("5 khoảng cách đầu tiên (TP1 đến TP2..TP6):")
    for j in range(1, 6):
        print(f"  TP1 -> TP{j+1}: {dist[0][j]:.1f}")
    print()
    print("Khoảng cách TP1 -> TP1:", dist[0][0], "(phải = 0)")