"""
Module đọc file TSPLIB (.tsp).

Hỗ trợ định dạng khoảng cách:
- EUC_2D: khoảng cách Euclidean 2D
- CEIL_2D: Euclidean làm tròn lên
- GEO: tọa độ địa lý
- ATT: khoảng cách kiểu ATT

Tác giả: [Tên bạn]
Ngày: [Ngày hôm nay]
"""

from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path


@dataclass
class TSPProblem:
    """
    Đại diện cho một bài toán TSP.

    Attributes:
        name: Tên bài toán (ví dụ: 'ch150')
        dimension: Số thành phố
        edge_weight_type: Loại khoảng cách ('EUC_2D', 'GEO', ...)
        coords: Danh sách tọa độ [(x1, y1), (x2, y2), ...]
        comment: Ghi chú từ file gốc
    """
    name: str
    dimension: int
    edge_weight_type: str
    coords: List[Tuple[float, float]]
    comment: str = ""


def load_tsp(filepath: str) -> TSPProblem:
    """
    Đọc file TSPLIB và trả về đối tượng TSPProblem.

    Args:
        filepath: Đường dẫn tới file .tsp

    Returns:
        TSPProblem chứa thông tin bài toán

    Raises:
        FileNotFoundError: Nếu file không tồn tại
        ValueError: Nếu file sai định dạng
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")

    name = ""
    dimension = 0
    edge_weight_type = ""
    comment = ""
    coords: List[Tuple[float, float]] = []

    reading_coords = False

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if reading_coords:
                if line == "EOF":
                    break
                parts = line.split()
                if len(parts) < 3:
                    continue
                x = float(parts[1])
                y = float(parts[2])
                coords.append((x, y))
                continue

            if line.startswith("NAME"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("DIMENSION"):
                dimension = int(line.split(":", 1)[1].strip())
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                edge_weight_type = line.split(":", 1)[1].strip()
            elif line.startswith("COMMENT"):
                comment = line.split(":", 1)[1].strip()
            elif line.startswith("NODE_COORD_SECTION"):
                reading_coords = True

    if dimension == 0:
        raise ValueError("File thiếu dòng DIMENSION")
    if not coords:
        raise ValueError("File thiếu phần NODE_COORD_SECTION")
    if len(coords) != dimension:
        raise ValueError(
            f"Số tọa độ ({len(coords)}) không khớp DIMENSION ({dimension})"
        )

    return TSPProblem(
        name=name,
        dimension=dimension,
        edge_weight_type=edge_weight_type,
        coords=coords,
        comment=comment,
    )


def print_problem_info(problem: TSPProblem) -> None:
    """In thông tin bài toán ra màn hình."""
    print("=" * 50)
    print(f"Tên bài toán:      {problem.name}")
    print(f"Số thành phố:      {problem.dimension}")
    print(f"Loại khoảng cách:  {problem.edge_weight_type}")
    print(f"Ghi chú:           {problem.comment}")
    print(f"Số tọa độ đọc được: {len(problem.coords)}")
    print(f"Tọa độ 3 thành phố đầu:")
    for i, (x, y) in enumerate(problem.coords[:3]):
        print(f"  Thành phố {i+1}: ({x}, {y})")
    print("=" * 50)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Cách dùng: python core/data_loader.py <đường_dẫn_file.tsp>")
        sys.exit(1)

    filepath = sys.argv[1]
    problem = load_tsp(filepath)
    print_problem_info(problem)