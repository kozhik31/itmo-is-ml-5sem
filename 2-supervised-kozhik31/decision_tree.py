from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
import numpy.typing as npt
import pandas as pd

@dataclass
class Node:
    is_leaf: bool = False
    loss: Optional[float] = None
    feature_idx: Optional[int] = None  # индекс фичи вместо имени
    separation: Optional[float] = None
    predict: Optional[float] = None
    samples: int = 0
    left: Optional["Node"] = None
    right: Optional["Node"] = None

    def __str__(self):
        return f"{self.loss=}, {self.feature_idx=}, {self.separation=}, {self.predict=}, {self.samples=}\n"


class DecisionTree:
    def __init__(self):
        self.root: Optional[Node] = None
        self.max_depth: int | None = None
        self.min_samples: int | None = None
        self.feature_names_: List[str] | None = None  # чтобы печатать красиво

    # SSE = sum(y^2) - sum(y)^2 / n; MSE = SSE / n
    @staticmethod
    def _sse(y: npt.NDArray[np.float64], s: float | None = None, s2: float | None = None) -> float:
        if s is None:
            s = float(y.sum())
        if s2 is None:
            s2 = float(np.dot(y, y))
        n = y.size
        if n == 0:
            return 0.0
        return s2 - (s * s) / n

    def _best_split_1d(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        min_samples: int
    ) -> Tuple[Optional[float], float]:
        """
        Возвращает (separation, loss). Если сплита нет — (None, +inf).
        Делает один sort и один линейный проход с префиксными суммами.
        """
        n = y.size
        if n < 2 * min_samples:
            return None, float("inf")

        order = np.argsort(x, kind="mergesort")  # stable
        xs = x[order]
        ys = y[order]

        # кандидаты — только там, где значение меняется
        diff = xs[1:] != xs[:-1]
        if not np.any(diff):
            return None, float("inf")

        # префиксы
        prefix_sum = np.cumsum(ys, dtype=np.float64)
        prefix_sqsum = np.cumsum(ys * ys, dtype=np.float64)

        # размеры левой/правой частей, если сплит после позиции i (берём i = 0..n-2)
        n_left = np.arange(1, n, dtype=np.int64)
        n_right = n - n_left

        # SSE_left(i): по префиксам
        s_left = prefix_sum[:-1]
        s2_left = prefix_sqsum[:-1]
        sse_left = s2_left - (s_left * s_left) / n_left

        # SSE_right(i): по суффиксам
        total_s = prefix_sum[-1]
        total_s2 = prefix_sqsum[-1]
        s_right = total_s - s_left
        s2_right = total_s2 - s2_left
        sse_right = s2_right - (s_right * s_right) / n_right

        # валидные индексы по min_samples + разным значениям
        valid = (n_left >= min_samples) & (n_right >= min_samples) & diff

        if not np.any(valid):
            return None, float("inf")

        # средняя ошибка = (SSE_L + SSE_R) / n
        loss_all = (sse_left + sse_right) / n
        # берём минимум только по валидным
        best_i = np.argmin(np.where(valid, loss_all, np.inf))
        best_loss = float(loss_all[best_i])

        # порог между xs[best_i] и xs[best_i+1]
        separation = float((xs[best_i] + xs[best_i + 1]) / 2.0)
        return separation, best_loss

    def _best_split(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> Tuple[Optional[int], Optional[float], float]:
        n_features = X.shape[1]
        best_feat = None
        best_sep = None
        best_loss = float("inf")

        for j in range(n_features):
            sep, loss = self._best_split_1d(X[:, j], y, self.min_samples)
            if sep is not None and loss < best_loss:
                best_feat = j
                best_sep = sep
                best_loss = loss

        return best_feat, best_sep, best_loss

    def _grow_tree(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64], depth: int = 0) -> Node:
        # критерии остановки
        if depth == self.max_depth or y.size < 2 * self.min_samples:
            return Node(is_leaf=True, samples=y.size, predict=float(y.mean()))

        feat, sep, loss = self._best_split(X, y)
        if sep is None or feat is None:
            return Node(is_leaf=True, samples=y.size, predict=float(y.mean()))

        # разбиение без pandas
        mask_left = X[:, feat] <= sep
        mask_right = ~mask_left

        left_X, right_X = X[mask_left], X[mask_right]
        left_y, right_y = y[mask_left], y[mask_right]

        # безопасность по min_samples
        if left_y.size < self.min_samples or right_y.size < self.min_samples:
            return Node(is_leaf=True, samples=y.size, predict=float(y.mean()))

        node = Node(is_leaf=False, samples=y.size, feature_idx=feat, separation=sep, loss=loss)
        node.left = self._grow_tree(left_X, left_y, depth + 1)
        node.right = self._grow_tree(right_X, right_y, depth + 1)
        return node

    def fit(self, X: pd.DataFrame, y: pd.Series, max_depth: int = 5, min_samples: int = 1):
        self.max_depth = max_depth
        self.min_samples = max(1, int(min_samples))
        self.feature_names_ = list(X.columns)

        X_np = X.to_numpy(dtype=np.float64, copy=False)
        y_np = y.to_numpy(dtype=np.float64, copy=False)

        self.root = self._grow_tree(X_np, y_np, depth=0)

    def show(self, node: Optional[Node] = None, depth: int = 0):
        if node is None:
            node = self.root
        if node is None:
            print("<empty tree>")
            return

        indent = "  " * depth
        if node.is_leaf:
            print(f"{indent}Leaf(samples={node.samples}, predict={node.predict:.4f})")
        else:
            fname = self.feature_names_[node.feature_idx] if self.feature_names_ else f"f{node.feature_idx}"
            print(f"{indent}Node(f={fname}[{node.feature_idx}] <= {node.separation:.6g}, "
                  f"samples={node.samples}, loss={node.loss:.6g})")
            self.show(node.left, depth + 1)
            self.show(node.right, depth + 1)

    def predict(self, X: pd.DataFrame | npt.NDArray[np.float64]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            Xn = X.to_numpy(dtype=np.float64, copy=False)
        else:
            Xn = np.asarray(X, dtype=np.float64)

        n = Xn.shape[0]
        out = np.empty(n, dtype=np.float64)

        for i in range(n):
            node = self.root
            while node and not node.is_leaf:
                if Xn[i, node.feature_idx] <= node.separation:
                    node = node.left
                else:
                    node = node.right
            out[i] = node.predict if node else np.nan
        return out
