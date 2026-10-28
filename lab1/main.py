import numpy as np
from typing import Callable
from pathlib import Path
import time
import matplotlib.pyplot as plt

def generate_square(n: int, xy_range: float) -> np.ndarray:
    return np.random.uniform(-xy_range, xy_range, size=(n, 2))

def generate_circle(n: int, r: float) -> np.ndarray:
    t = np.random.uniform(0.0, 1.0, size=n)
    x = r * np.cos(2 * np.pi * t)
    y = r * np.sin(2 * np.pi * t)
    return np.column_stack((x,y))

def generate_line(n: int, a: tuple[float, float], b: tuple[float, float], x_range: float) -> np.ndarray:
    ax, ay = a
    bx, by = b
    x = np.random.uniform(-x_range, x_range, size=n)
    slope = (ay - by) / (ax - bx)
    y = ay + slope * (x - ax)
    return np.column_stack((x, y))

def load_or_generate_points(dir_name: str, a, b) -> dict[str, np.ndarray]:
    dir = Path(dir_name)
    dir.mkdir(exist_ok=True)
    points = {}
    generators = {
        'a': lambda: generate_square(10**5, 1000),
        'b': lambda: generate_square(10**5, 1e14),
        'c': lambda: generate_circle(1000, r=100),
        'd': lambda: generate_line(1000, a, b, 1000)
    }

    for name, gen in generators.items():
        file = dir / f'points_{name}.npy'
        if file.exists():
            points[name] = np.load(file)
        else:
            points[name] = gen()
            np.save(file, points[name])

    return points

# |ax ay 1|
# |bx by 1|
# |cx cy 1|
def cross3x3(ax: float, ay: float, bx: float, by: float, cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    return ax*by + ay*cx + bx*cy - by*cx - cy*ax - ay*bx

def cross3x3_numpy(ax: float, ay: float, bx: float, by: float, cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    n = len(cx)
    ones = np.ones(n)
    matrix = np.stack([
        np.column_stack([np.full(n, ax), np.full(n, ay), ones]),
        np.column_stack([np.full(n, bx), np.full(n, by), ones]),
        np.column_stack([           cx,             cy,  ones]),
    ], axis=1)
    return np.linalg.det(matrix)

# |ax-cx ay-cy|
# |bx-cx by-cy|
def cross2x2(ax: float, ay: float, bx: float, by: float, cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    return (ax - cx) * (by - cy) - (ay - cy) * (bx - cx)

def cross2x2_numpy(ax: float, ay: float, bx: float, by: float, cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    n = len(cx)
    matrix = np.stack([
        np.column_stack([ax - cx, ay - cy]),
        np.column_stack([bx - cx, by - cy]),
    ], axis=1)
    return np.linalg.det(matrix)


def classify_points(points: np.ndarray, a: tuple[float, float], b: tuple[float, float],
                    epsilon: float, cross: Callable[[float, float, float, float, np.ndarray, np.ndarray], np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cx = points[:, 0]
    cy = points[:, 1]
    ax, ay = a
    bx, by = b

    dets = cross(ax, ay, bx, by, cx, cy)
    left = points[dets > epsilon]
    right = points[dets < -epsilon]
    collinear = points[abs(dets) < epsilon]

    return left, right, collinear

def compare(points: dict[str, np.ndarray], methods: dict[str, Callable], 
            epsilons: list[float], file_name: str, a: tuple[float, float], b: tuple[float, float]) -> None:
    file = Path(file_name)
    header = f'{'Dtype':<7} | {'Method':<10} | {'Dataset':<8} | {'Epsilon':<7} | {'Time':<8} | {'Left':<6} | {'Right':<6} | {'Collinear':<6}'
    results = [header]

    for dtype in [np.float64, np.float32]:
        dtype_name = np.dtype(dtype).name

        for method_name, func in methods.items():
            for dataset_name, dataset in points.items():
                dataset_typed = dataset.astype(dtype)
                    
                for epsilon in epsilons:
                    start_time = time.perf_counter()
                    left, right, collinear = classify_points(dataset_typed, a, b, epsilon, func)
                    end_time = time.perf_counter()
                    elapsed_time = end_time - start_time

                    row = f'{dtype_name:<7} | {method_name:<10} | {dataset_name:<8} | {epsilon:<7} | {elapsed_time:<5.4f} s | {len(left):<6} | {len(right):<6} | {len(collinear):<6} '
                    results.append(row)
            results.append(len(header)*'-')
        results.append(len(header)*'=')

    file.write_text('\n'.join(results), encoding='utf-8')

def plot(points: dict[str, np.ndarray], a: tuple[float, float], b: tuple[float, float], cross_func: Callable, epsilon: float, save:bool, output_name:str=''):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i, (name, data) in enumerate(points.items()):
        left, right, collinear = classify_points(data, a, b, epsilon, cross_func)
        axes[0, i].scatter(data[:,0], data[:,1], s=0.5, c='blue', alpha=0.5)
        axes[1, i].scatter(collinear[:,0], collinear[:,1], s=0.5, c='black', alpha=0.5)
        axes[1, i].scatter(left[:,0], left[:,1], s=0.5, c='green', alpha=0.5)
        axes[1, i].scatter(right[:,0], right[:,1], s=0.5, c='red', alpha=0.5)

        axes[0, i].set_aspect('equal')
        axes[0, i].set_box_aspect(1)
        axes[1, i].set_aspect('equal')
        axes[1, i].set_box_aspect(1)
    
    plt.tight_layout()
    plt.show()
    if save:
        out_path = Path(output_name)    
        #out_path.mkdir(exist_ok=True)
        fig.savefig(out_path)


if __name__ == '__main__':
    a = (-1.0, 0.0)
    b = (1.0, 0.1)
    points = load_or_generate_points('data', a, b)
    methods = {'3x3': cross3x3, '2x2': cross2x2, '3x3 numpy': cross3x3_numpy, '2x2 numpy': cross2x2_numpy}
    epsilons = [1e-6, 1e-9, 1e-12]
    #compare(points, methods, epsilons, 'results.txt', a, b)
    plot(points, a, b, cross2x2, 1e-12, save=False, output_name='plots/2x2e-12.png')
