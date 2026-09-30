"""
Giao diện đồ họa (GUI) cho phần mềm giải bài toán TSP.

Sử dụng Tkinter (thư viện có sẵn của Python).

Chức năng:
- Chọn thuật toán: GA, SA, NN
- Điều chỉnh tham số (slider + entry)
- Vẽ bản đồ thành phố trên canvas
- Chạy thuật toán, hiển thị hành trình tốt nhất
- Lưu/Mở bản đồ dưới dạng JSON
- Nạp file TSPLIB (.tsp)
- Xem kết quả và thống kê

Tác giả: [Tên bạn]
Ngày: [Ngày hôm nay]
"""

import sys
import json
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

# Thêm thư mục gốc vào sys.path
sys.path.insert(0, str(Path(__file__).parent))

from core.data_loader import load_tsp
from core.distance import compute_distance_matrix
from core.ga import run_ga
from core.algorithms.nearest_neighbor import nearest_neighbor
from core.algorithms.simulated_annealing import simulated_annealing


KNOWN_OPTIMA = {
    "berlin52": 7542, "eil51": 426, "ch150": 6528,
    "kroA100": 21282, "tsp225": 3919, "pr439": 107217,
    "pcb442": 50778,
}


class TSPApp:
    """Ứng dụng chính."""

    def __init__(self, root):
        self.root = root
        self.root.title("Phần mềm giải bài toán TSP - GA / SA / NN")
        self.root.geometry("1350x870")

        # Trạng thái
        self.coords = None
        self.dist = None
        self.problem_name = ""
        self.best_tour = None
        self.best_length = None
        self.optimal = None
        self.running = False
        self.canvas_offset = (0, 0)
        self.canvas_scale = 1.0

        # Build UI
        self._build_menu()
        self._build_layout()

    # ============================================================
    # MENU
    # ============================================================
    def _build_menu(self):
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Nạp file .tsp", command=self.load_tsp_file)
        file_menu.add_command(label="Mở bản đồ (JSON)", command=self.open_map)
        file_menu.add_command(label="Lưu bản đồ (JSON)", command=self.save_map)
        file_menu.add_separator()
        file_menu.add_command(label="Xuất ảnh canvas", command=self.export_canvas_image)
        file_menu.add_separator()
        file_menu.add_command(label="Thoát", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Hướng dẫn", command=self.show_help)
        help_menu.add_command(label="Giới thiệu", command=self.show_about)
        menubar.add_cascade(label="Trợ giúp", menu=help_menu)

        self.root.config(menu=menubar)

    # ============================================================
    # LAYOUT
    # ============================================================
    def _build_layout(self):
        # Container chính
        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # === PANEL TRÁI ===
        left = tk.LabelFrame(main, text="Bảng điều khiển", width=320)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left.pack_propagate(False)
        self._build_left_panel(left)

        # === PANEL PHẢI ===
        right = tk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_right_panel(right)

        # === STATUS BAR ===
        self.status_var = tk.StringVar(value="Sẵn sàng. Hãy nạp file .tsp hoặc mở bản đồ JSON.")
        status = tk.Label(
            self.root, textvariable=self.status_var,
            bd=1, relief=tk.SUNKEN, anchor=tk.W, padx=5,
        )
        status.pack(side=tk.BOTTOM, fill=tk.X)

    def _build_left_panel(self, parent):
        # --- Chọn thuật toán ---
        frame_algo = tk.LabelFrame(parent, text="Thuật toán")
        frame_algo.pack(fill=tk.X, padx=5, pady=5)

        self.algo_var = tk.StringVar(value="GA")
        for algo, label in [("GA", "Genetic Algorithm"), ("SA", "Simulated Annealing"), ("NN", "Nearest Neighbor")]:
            tk.Radiobutton(
                frame_algo, text=label, variable=self.algo_var, value=algo,
                anchor=tk.W,
            ).pack(fill=tk.X, padx=5)

        # --- Tham số GA ---
        frame_ga = tk.LabelFrame(parent, text="Tham số GA")
        frame_ga.pack(fill=tk.X, padx=5, pady=5)

        self.ga_pop = self._add_slider(frame_ga, "Quần thể:", 10, 200, 50)
        self.ga_gen = self._add_slider(frame_ga, "Số thế hệ:", 100, 2000, 500)
        self.ga_mut = self._add_slider(frame_ga, "Đột biến:", 0.01, 0.5, 0.1, is_float=True)

        # --- Tham số SA ---
        frame_sa = tk.LabelFrame(parent, text="Tham số SA")
        frame_sa.pack(fill=tk.X, padx=5, pady=5)

        self.sa_iter = self._add_slider(frame_sa, "Vòng lặp:", 10000, 500000, 100000)
        self.sa_temp = self._add_slider(frame_sa, "Nhiệt độ:", 100, 5000, 1000)

        # --- Nút điều khiển ---
        frame_btn = tk.Frame(parent)
        frame_btn.pack(fill=tk.X, padx=5, pady=10)

        self.btn_run = tk.Button(
            frame_btn, text="▶ CHẠY THUẬT TOÁN",
            bg="#4CAF50", fg="white", font=("Arial", 10, "bold"),
            command=self.run_algorithm, height=2,
        )
        self.btn_run.pack(fill=tk.X, pady=2)

        self.btn_clear = tk.Button(
            frame_btn, text="🗑 Xóa bản đồ",
            command=self.clear_canvas, height=1,
        )
        self.btn_clear.pack(fill=tk.X, pady=2)

        self.btn_save = tk.Button(
            frame_btn, text="💾 Lưu bản đồ (JSON)",
            command=self.save_map, height=1,
        )
        self.btn_save.pack(fill=tk.X, pady=2)

        self.btn_open = tk.Button(
            frame_btn, text="📂 Mở bản đồ (JSON)",
            command=self.open_map, height=1,
        )
        self.btn_open.pack(fill=tk.X, pady=2)

        self.btn_load = tk.Button(
            frame_btn, text="📁 Nạp file .tsp",
            command=self.load_tsp_file, height=1,
        )
        self.btn_load.pack(fill=tk.X, pady=2)

        self.btn_reset = tk.Button(
            frame_btn, text="↺ Huỷ (về dataset)",
            command=self.reset_to_dataset, height=1,
        )
        self.btn_reset.pack(fill=tk.X, pady=2)

        # --- Ghi chú ---
        note = tk.Label(
            parent,
            text="Ghi chú:\n- Click trái lên canvas: thêm TP\n- Click phải: xóa TP gần nhất\n- Nạp file .tsp để dùng dữ liệu chuẩn",
            justify=tk.LEFT, anchor=tk.W, fg="#555",
            font=("Arial", 8),
        )
        note.pack(fill=tk.X, padx=5, pady=5)

    def _add_slider(self, parent, label, min_val, max_val, default, is_float=False):
        """
        Thêm một slider với label, entry và giá trị hiển thị.

        Người dùng có thể:
        - Kéo slider để chọn giá trị.
        - Nhập trực tiếp vào ô Entry (nhấn Enter hoặc click ra ngoài để chốt).
        """
        frame = tk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=3)

        # Label bên trái
        tk.Label(frame, text=label, width=11, anchor=tk.W).pack(side=tk.LEFT)

        # Biến lưu giá trị
        var = tk.DoubleVar(value=default) if is_float else tk.IntVar(value=default)

        # --- Entry bên phải (nhập trực tiếp) ---
        entry = tk.Entry(frame, width=8, justify=tk.RIGHT)
        entry.pack(side=tk.RIGHT, padx=(3, 0))

        # Khởi tạo giá trị cho Entry
        if is_float:
            entry.insert(0, f"{float(default):.3f}")
        else:
            entry.insert(0, str(int(default)))

        # --- Slider ở giữa ---
        def on_slider_change(v):
            # Cập nhật Entry khi kéo slider
            entry.delete(0, tk.END)
            if is_float:
                entry.insert(0, f"{float(v):.3f}")
            else:
                entry.insert(0, str(int(float(v))))

        scale = tk.Scale(
            frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL,
            variable=var, showvalue=False, command=on_slider_change,
            length=110, resolution=0.001 if is_float else 1,
        )
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)

        # --- Xử lý khi người dùng nhập vào Entry ---
        def on_entry_commit(event=None):
            try:
                raw = entry.get().strip()
                if raw == "":
                    raise ValueError("empty")
                val = float(raw) if is_float else int(float(raw))

                # Clamp giá trị vào khoảng hợp lệ
                if val < min_val:
                    val = min_val
                elif val > max_val:
                    val = max_val

                # Cập nhật biến (sẽ trigger on_slider_change)
                var.set(val)

                # Định dạng lại Entry cho đẹp
                entry.delete(0, tk.END)
                if is_float:
                    entry.insert(0, f"{float(val):.3f}")
                else:
                    entry.insert(0, str(int(val)))
            except (ValueError, tk.TclError):
                # Nếu nhập sai → khôi phục giá trị hiện tại
                val = var.get()
                entry.delete(0, tk.END)
                if is_float:
                    entry.insert(0, f"{float(val):.3f}")
                else:
                    entry.insert(0, str(int(float(val))))

        # Nhấn Enter hoặc click ra ngoài → chốt giá trị
        entry.bind("<Return>", on_entry_commit)
        entry.bind("<FocusOut>", on_entry_commit)

        return var

    def _build_right_panel(self, parent):
        # --- Canvas ---
        frame_canvas = tk.LabelFrame(parent, text="Bản đồ thành phố (click trái để thêm, click phải để xóa)")
        frame_canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            frame_canvas, bg="white", highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Bind sự kiện chuột
        self.canvas.bind("<Button-1>", self.on_canvas_left_click)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)
        self.canvas.bind("<Configure>", lambda e: self.redraw_canvas())

        # --- Notebook kết quả ---
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=False, pady=(5, 0))

        # Tab 1: Kết quả
        tab1 = tk.Frame(notebook)
        notebook.add(tab1, text="Kết quả")

        self.result_text = tk.Text(
            tab1, height=10, font=("Consolas", 10), wrap=tk.WORD,
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # Tab 2: Thống kê
        tab2 = tk.Frame(notebook)
        notebook.add(tab2, text="Thống kê")

        self.stats_text = tk.Text(
            tab2, height=10, font=("Consolas", 10), wrap=tk.WORD,
        )
        self.stats_text.pack(fill=tk.BOTH, expand=True)

    # ============================================================
    # CANVAS
    # ============================================================
    def get_canvas_size(self):
        return self.canvas.winfo_width(), self.canvas.winfo_height()

    def compute_scale(self):
        """Tính scale và offset để vẽ coords vừa canvas."""
        if self.coords is None or len(self.coords) == 0:
            return 1.0, 0, 0

        w, h = self.get_canvas_size()
        if w < 10 or h < 10:
            w, h = 800, 500

        margin = 30
        xs = self.coords[:, 0]
        ys = self.coords[:, 1]
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        x_range = max(x_max - x_min, 1e-6)
        y_range = max(y_max - y_min, 1e-6)

        scale = min(
            (w - 2 * margin) / x_range,
            (h - 2 * margin) / y_range,
        )

        offset_x = margin + (w - 2 * margin - x_range * scale) / 2 - x_min * scale
        offset_y = margin + (h - 2 * margin - y_range * scale) / 2 - y_min * scale

        return scale, offset_x, offset_y

    def coord_to_canvas(self, x, y, scale, offset_x, offset_y):
        """Chuyển tọa độ thực sang tọa độ canvas (đảo trục y)."""
        cx = x * scale + offset_x
        w, h = self.get_canvas_size()
        cy = h - (y * scale + offset_y)
        return cx, cy

    def canvas_to_coord(self, cx, cy, scale, offset_x, offset_y):
        """Chuyển tọa độ canvas sang tọa độ thực."""
        w, h = self.get_canvas_size()
        x = (cx - offset_x) / scale
        y = (h - cy - offset_y) / scale
        return x, y

    def redraw_canvas(self):
        """Vẽ lại toàn bộ canvas."""
        self.canvas.delete("all")

        if self.coords is None or len(self.coords) == 0:
            return

        scale, offset_x, offset_y = self.compute_scale()
        self.canvas_scale = scale
        self.canvas_offset = (offset_x, offset_y)

        # Vẽ cạnh (tour) nếu có
        if self.best_tour is not None and len(self.best_tour) > 1:
            tour = self.best_tour
            n = len(tour)
            for i in range(n):
                a = tour[i]
                b = tour[(i + 1) % n]
                x1, y1 = self.coord_to_canvas(*self.coords[a], scale, offset_x, offset_y)
                x2, y2 = self.coord_to_canvas(*self.coords[b], scale, offset_x, offset_y)
                self.canvas.create_line(x1, y1, x2, y2, fill="#2196F3", width=1.5, tags="edge")

        # Vẽ các thành phố
        for i, (x, y) in enumerate(self.coords):
            cx, cy = self.coord_to_canvas(x, y, scale, offset_x, offset_y)
            self.canvas.create_oval(
                cx - 4, cy - 4, cx + 4, cy + 4,
                fill="#F44336", outline="#B71C1C", width=1,
                tags=("city", f"city_{i}"),
            )
            self.canvas.create_text(
                cx + 8, cy - 8, text=str(i),
                font=("Arial", 7), fill="#333", anchor=tk.W,
                tags=("label", f"label_{i}"),
            )

        # Đánh dấu thành phố xuất phát
        if self.best_tour is not None and len(self.best_tour) > 0:
            start = self.best_tour[0]
            x, y = self.coords[start]
            cx, cy = self.coord_to_canvas(x, y, scale, offset_x, offset_y)
            self.canvas.create_oval(
                cx - 8, cy - 8, cx + 8, cy + 8,
                outline="#4CAF50", width=3, tags="start",
            )

    def on_canvas_left_click(self, event):
        """Thêm thành phố khi click chuột trái."""
        if self.running:
            return

        if self.coords is None:
            self.coords = np.empty((0, 2))
            self.problem_name = "custom"

        scale, offset_x, offset_y = self.compute_scale()
        x, y = self.canvas_to_coord(event.x, event.y, scale, offset_x, offset_y)

        self.coords = np.vstack([self.coords, [x, y]])

        self.best_tour = None
        self.best_length = None
        self.dist = None
        self.optimal = None

        self.redraw_canvas()
        self.status_var.set(f"Đã thêm thành phố {len(self.coords) - 1}. Tổng: {len(self.coords)}")

    def on_canvas_right_click(self, event):
        """Xóa thành phố gần nhất khi click chuột phải."""
        if self.running or self.coords is None or len(self.coords) == 0:
            return

        scale, offset_x, offset_y = self.compute_scale()

        min_dist = float("inf")
        min_idx = -1
        for i, (x, y) in enumerate(self.coords):
            cx, cy = self.coord_to_canvas(x, y, scale, offset_x, offset_y)
            d = (cx - event.x) ** 2 + (cy - event.y) ** 2
            if d < min_dist:
                min_dist = d
                min_idx = i

        if min_dist < 15 ** 2 and min_idx >= 0:
            self.coords = np.delete(self.coords, min_idx, axis=0)
            self.best_tour = None
            self.best_length = None
            self.dist = None
            self.optimal = None
            self.redraw_canvas()
            self.status_var.set(f"Đã xóa thành phố {min_idx}. Tổng: {len(self.coords)}")

    # ============================================================
    # CHỨC NĂNG NẠP / LƯU / XÓA
    # ============================================================
    def load_tsp_file(self):
        """Nạp file .tsp."""
        filepath = filedialog.askopenfilename(
            title="Chọn file TSPLIB",
            filetypes=[("TSP files", "*.tsp"), ("All files", "*.*")],
        )
        if not filepath:
            return

        try:
            problem = load_tsp(filepath)
            self.coords = np.array(problem.coords, dtype=np.float64)
            self.problem_name = problem.name
            self.dist = compute_distance_matrix(self.coords, problem.edge_weight_type)
            self.optimal = KNOWN_OPTIMA.get(problem.name)
            self.best_tour = None
            self.best_length = None

            self.redraw_canvas()
            self.status_var.set(
                f"Đã nạp: {problem.name} ({problem.dimension} thành phố) | "
                f"Tối ưu: {self.optimal if self.optimal else 'N/A'}"
            )
            self.log_result(f"Đã nạp file: {filepath}\n"
                            f"Tên: {problem.name}\n"
                            f"Số thành phố: {problem.dimension}\n"
                            f"Loại: {problem.edge_weight_type}\n"
                            f"Tối ưu: {self.optimal if self.optimal else 'Không có'}\n")

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể nạp file:\n{e}")

    def save_map(self):
        """Lưu bản đồ hiện tại ra file JSON."""
        if self.coords is None or len(self.coords) == 0:
            messagebox.showwarning("Cảnh báo", "Chưa có dữ liệu để lưu.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Lưu bản đồ",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
        )
        if not filepath:
            return

        data = {
            "name": self.problem_name or "custom",
            "coords": self.coords.tolist(),
            "best_tour": self.best_tour.tolist() if self.best_tour is not None else None,
            "best_length": float(self.best_length) if self.best_length is not None else None,
            "optimal": self.optimal,
        }

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.status_var.set(f"Đã lưu bản đồ: {filepath}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu:\n{e}")

    def open_map(self):
        """Mở bản đồ từ file JSON."""
        filepath = filedialog.askopenfilename(
            title="Mở bản đồ",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not filepath:
            return

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.coords = np.array(data["coords"], dtype=np.float64)
            self.problem_name = data.get("name", "custom")
            self.dist = compute_distance_matrix(self.coords, "EUC_2D")
            self.optimal = data.get("optimal")

            if data.get("best_tour"):
                self.best_tour = np.array(data["best_tour"], dtype=np.int32)
                self.best_length = data.get("best_length")
            else:
                self.best_tour = None
                self.best_length = None

            self.redraw_canvas()
            self.status_var.set(
                f"Đã mở bản đồ: {self.problem_name} ({len(self.coords)} thành phố)"
            )
            self.log_result(f"Đã mở bản đồ: {filepath}\n"
                            f"Tên: {self.problem_name}\n"
                            f"Số thành phố: {len(self.coords)}\n"
                            f"Tối ưu: {self.optimal if self.optimal else 'Không có'}\n")

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể mở file:\n{e}")

    def clear_canvas(self):
        """Xóa toàn bộ thành phố."""
        if messagebox.askyesno("Xác nhận", "Xóa toàn bộ bản đồ?"):
            self.coords = None
            self.dist = None
            self.best_tour = None
            self.best_length = None
            self.optimal = None
            self.problem_name = ""
            self.redraw_canvas()
            self.result_text.delete("1.0", tk.END)
            self.stats_text.delete("1.0", tk.END)
            self.status_var.set("Đã xóa bản đồ.")

    def reset_to_dataset(self):
        """Huỷ về dataset ban đầu."""
        if not self.problem_name or self.coords is None:
            self.clear_canvas()
            return

        self.best_tour = None
        self.best_length = None
        self.redraw_canvas()
        self.status_var.set(f"Đã huỷ về dataset gốc: {self.problem_name}")

    def export_canvas_image(self):
        """Xuất canvas ra file ảnh (PostScript)."""
        if self.coords is None:
            messagebox.showwarning("Cảnh báo", "Chưa có gì để xuất.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Xuất ảnh",
            defaultextension=".ps",
            filetypes=[("PostScript", "*.ps")],
        )
        if not filepath:
            return

        try:
            self.canvas.postscript(file=filepath, colormode="color")
            self.status_var.set(f"Đã xuất ảnh: {filepath}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể xuất ảnh:\n{e}")

    # ============================================================
    # CHẠY THUẬT TOÁN
    # ============================================================
    def run_algorithm(self):
        """Chạy thuật toán đã chọn."""
        if self.running:
            messagebox.showwarning("Cảnh báo", "Đang chạy thuật toán. Vui lòng đợi.")
            return

        if self.coords is None or len(self.coords) < 3:
            messagebox.showwarning("Cảnh báo", "Cần ít nhất 3 thành phố. Hãy nạp file hoặc click lên canvas.")
            return

        if self.dist is None:
            self.dist = compute_distance_matrix(self.coords, "EUC_2D")

        algo = self.algo_var.get()
        self.running = True
        self.btn_run.config(state=tk.DISABLED, text="Đang chạy...")
        self.status_var.set(f"Đang chạy {algo}...")
        self.log_result(f"\n{'=' * 60}\nBẮT ĐẦU CHẠY {algo}\n{'=' * 60}\n")

        thread = threading.Thread(target=self._run_thread, args=(algo,), daemon=True)
        thread.start()

    def _run_thread(self, algo):
        """Chạy thuật toán trong thread riêng."""
        t0 = time.time()
        try:
            if algo == "NN":
                result = nearest_neighbor(self.dist, n_starts=None, verbose=False)
                best_tour = result.best_tour
                best_length = result.best_length
                history = [best_length]
                extra = f"Số điểm xuất phát: {len(result.all_lengths)}\n"

            elif algo == "SA":
                nn_result = nearest_neighbor(self.dist, n_starts=None, verbose=False)
                result = simulated_annealing(
                    self.dist,
                    initial_tour=nn_result.best_tour,
                    max_iterations=int(self.sa_iter.get()),
                    initial_temp=float(self.sa_temp.get()),
                    final_temp=0.01,
                    cooling_rate=0.9999,
                    move_type="inversion",
                    local_search_every=1000,
                    local_search_iter=50,
                    seed=None,
                    verbose=False,
                )
                best_tour = result.best_tour
                best_length = result.best_length
                history = result.history
                extra = (
                    f"Số vòng lặp: {result.iterations_run}\n"
                    f"Chấp nhận: {result.accepted_moves}\n"
                    f"Từ chối: {result.rejected_moves}\n"
                )

            else:  # GA
                result = run_ga(
                    dist=self.dist,
                    pop_size=int(self.ga_pop.get()),
                    generations=int(self.ga_gen.get()),
                    mutation_rate=float(self.ga_mut.get()),
                    elite_size=5,
                    tournament_k=3,
                    crossover_type="ox",
                    mutation_type="inversion",
                    use_2opt=True,
                    two_opt_iterations=20,
                    two_opt_every=1,
                    seed=None,
                    verbose=False,
                )
                best_tour = result.best_tour
                best_length = result.best_length
                history = result.history
                extra = (
                    f"Quần thể: {int(self.ga_pop.get())}\n"
                    f"Thế hệ: {int(self.ga_gen.get())}\n"
                    f"Đột biến: {float(self.ga_mut.get()):.3f}\n"
                )

            elapsed = time.time() - t0
            self.root.after(0, self._on_run_complete, algo, best_tour, best_length, history, elapsed, extra)

        except Exception as e:
            self.root.after(0, self._on_run_error, str(e))

    def _on_run_complete(self, algo, best_tour, best_length, history, elapsed, extra):
        """Callback khi chạy xong."""
        self.best_tour = best_tour
        self.best_length = best_length

        gap_str = "Không có đáp án tối ưu"
        if self.optimal:
            gap = 100 * (best_length - self.optimal) / self.optimal
            gap_str = f"{gap:.3f}%"

        self.log_result(
            f"\n--- KẾT QUẢ {algo} ---\n"
            f"Best length:  {best_length:.2f}\n"
            f"Thời gian:    {elapsed:.3f} giây\n"
            f"Sai số:       {gap_str}\n"
        )

        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert(tk.END, "=" * 50 + "\n")
        self.stats_text.insert(tk.END, f"THỐNG KÊ {algo}\n")
        self.stats_text.insert(tk.END, "=" * 50 + "\n\n")
        self.stats_text.insert(tk.END, f"Thuật toán: {algo}\n")
        self.stats_text.insert(tk.END, f"Số thành phố: {len(self.coords)}\n")
        self.stats_text.insert(tk.END, f"Best length: {best_length:.2f}\n")
        self.stats_text.insert(tk.END, f"Thời gian: {elapsed:.3f} giây\n")
        self.stats_text.insert(tk.END, f"Tối ưu: {self.optimal if self.optimal else 'N/A'}\n")
        self.stats_text.insert(tk.END, f"Sai số: {gap_str}\n\n")
        self.stats_text.insert(tk.END, extra)

        self.redraw_canvas()

        self.running = False
        self.btn_run.config(state=tk.NORMAL, text="▶ CHẠY THUẬT TOÁN")
        self.status_var.set(f"Hoàn thành {algo}. Best = {best_length:.2f} ({gap_str})")

    def _on_run_error(self, error_msg):
        """Callback khi có lỗi."""
        self.running = False
        self.btn_run.config(state=tk.NORMAL, text="▶ CHẠY THUẬT TOÁN")
        self.status_var.set("Lỗi khi chạy thuật toán.")
        messagebox.showerror("Lỗi", f"Đã xảy ra lỗi:\n{error_msg}")

    def log_result(self, msg):
        """Ghi vào tab kết quả."""
        self.result_text.insert(tk.END, msg)
        self.result_text.see(tk.END)

    # ============================================================
    # TRỢ GIÚP
    # ============================================================
    def show_help(self):
        messagebox.showinfo(
            "Hướng dẫn",
            "1. Nạp file .tsp hoặc mở bản đồ JSON.\n"
            "2. Hoặc click lên canvas để thêm thành phố.\n"
            "3. Chọn thuật toán (GA / SA / NN).\n"
            "4. Điều chỉnh tham số bằng slider hoặc ô nhập.\n"
            "5. Bấm 'CHẠY THUẬT TOÁN'.\n\n"
            "Lưu ý:\n"
            "- Click trái: thêm thành phố.\n"
            "- Click phải: xóa thành phố gần nhất.\n"
            "- Lưu/Mở: định dạng JSON."
        )

    def show_about(self):
        messagebox.showinfo(
            "Giới thiệu",
            "Phần mềm giải bài toán Người du lịch (TSP)\n"
            "sử dụng 3 thuật toán:\n"
            "- Genetic Algorithm (GA)\n"
            "- Simulated Annealing (SA)\n"
            "- Nearest Neighbor (NN)\n\n"
            "Tác giả: [Tên bạn]\n"
            "Năm: 2026"
        )


def main():
    root = tk.Tk()
    app = TSPApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()