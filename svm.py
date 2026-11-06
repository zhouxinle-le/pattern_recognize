import argparse
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from quadprog import solve_qp
from scipy.ndimage import label
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parent
IMAGE = ROOT / '00000341.jpg'
HOLDOUT_FILE = ROOT / 'data' / 'dot_20261106_215524.npz'
POINT_FILES = [
    ROOT / 'data' / name for name in (
        'dot_20261104_194615.npz',
        'dot_20261106_202155.npz',
        'box_20261106_195341.npz',
        'box_20261106_201201.npz',
    )
]
C0 = 2.0
GAMMA = 0.6
METHODS = ('qp', 'dual', 'rbf')
PANEL_SIZE = 4.425  # 英寸

def sample_grid(xy, y_points):
    # 合并框选数据后，每个 50 × 50 网格内，每类只保留最靠近格子中心的点。
    grid = xy // 50
    center = (grid + 0.5) * 50
    order = np.argsort(np.sum((xy - center) ** 2, axis=1), kind='stable')
    keys = np.column_stack((grid, y_points))
    _, indices = np.unique(keys[order], axis=0, return_index=True)
    return order[indices]


def read_points(files):
    coords, labels, is_dense = [], [], []

    for filename in files:
        with np.load(filename) as data:
            xy, tags = data['xy'], data['tag']
        coords.append(xy)
        labels.append(np.where(tags == 'B', 1.0, -1.0))
        is_dense.append(np.full(len(xy), len(xy) > 1000))

    xy = np.concatenate(coords)
    y_points = np.concatenate(labels)
    dense = np.concatenate(is_dense)

    sparse_indices = np.flatnonzero(~dense)
    dense_indices = np.flatnonzero(dense)
    sampled = sample_grid(xy[dense], y_points[dense])
    selected = np.r_[sparse_indices, dense_indices[sampled]]
    train_xy, indices = np.unique(xy[selected], axis=0, return_index=True)
    return train_xy, y_points[selected[indices]], xy


def pixel_features(image, train_xy):
    lab = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    light = lab[:, :, 0]
    channels = [cv2.blur(lab, (5, 5))]
    for size in (3, 11):
        mean = cv2.blur(light, (size, size))
        variance = cv2.blur(light * light, (size, size)) - mean * mean
        texture = np.log1p(np.sqrt(np.maximum(variance, 0)))
        channels.append(texture[:, :, None])

    features = np.concatenate(channels, axis=2)
    train = features[train_xy[:, 1], train_xy[:, 0]]
    features -= train.mean(axis=0)
    features /= np.maximum(train.std(axis=0), 1e-6)

    h, width = image.shape[:2]
    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, h))
    return np.dstack((features, 0.65 * x, 0.65 * y))


def fit_primal_qp(x_points, y_points):
    m, d = x_points.shape
    G = np.zeros((d + 1 + m, d + 1 + m), dtype=float)
    G[:d, :d] = np.eye(d)
    # quadprog 需要正定矩阵，沿用原代码的微小正则项。
    G[d, d] = 1e-6
    G[d+1:, d+1:] = 1e-8 * np.eye(m)

    a = np.zeros(d + 1 + m, dtype=float)
    a[d+1:] = -C0

    C_T = np.zeros((2 * m, d + 1 + m))
    C_T[:m, :d] = x_points * y_points[:, None]
    C_T[:m, d] = y_points
    C_T[:m, d+1:] = np.eye(m)
    C_T[m:, d+1:] = np.eye(m)

    b = np.r_[np.ones(m), np.zeros(m)]
    sol = solve_qp(G, a, C_T.T, b, meq=0)
    return sol[0][:d], sol[0][d]


def fit_dual(kernel, y_points):
    m = len(y_points)
    G = kernel * np.outer(y_points, y_points)
    G += 1e-8 * np.eye(m)
    a = np.ones(m, dtype=float)

    C_T = np.zeros((2 * m + 1, m))
    C_T[0, :] = y_points           # y^T * alpha = 0
    C_T[1:1+m, :] = np.eye(m)     # alpha >= 0
    C_T[1+m:, :] = -np.eye(m)     # alpha <= C0

    b = np.r_[0.0, np.zeros(m), -C0 * np.ones(m)]
    sol = solve_qp(G, a, C_T.T, b, meq=1)

    alpha = sol[0]
    beta = alpha * y_points
    offsets = y_points - kernel @ beta
    at_zero = alpha <= 1e-5
    at_cap = alpha >= C0 - 1e-5
    free = ~(at_zero | at_cap)
    if np.any(free):
        b_val = offsets[free].mean()
    else:
        # 没有自由支持向量时，从 KKT 条件给出的偏置区间取中点。
        positive = y_points > 0
        lower = offsets[(positive & at_zero) | (~positive & at_cap)].max()
        upper = offsets[(positive & at_cap) | (~positive & at_zero)].min()
        b_val = (lower + upper) / 2

    return beta, b_val


def fit_linear_dual(x_points, y_points):
    beta, b_val = fit_dual(x_points @ x_points.T, y_points)
    return beta @ x_points, b_val, beta


def rbf_kernel(x, z):
    distances = cdist(x, z, metric='sqeuclidean')
    return np.exp(-GAMMA * distances)


def fit_rbf(x_points, y_points):
    beta, b_val = fit_dual(rbf_kernel(x_points, x_points), y_points)
    sv = np.abs(beta) > 1e-8
    return x_points[sv], beta[sv], b_val


def predict_rbf(img_flat, support, coefficients, b_val):
    Z = np.empty(len(img_flat), dtype=float)

    # 分批计算像素和支持向量之间的核值。
    for start in range(0, len(img_flat), 8192):
        batch = slice(start, start + 8192)
        Z[batch] = rbf_kernel(img_flat[batch], support) @ coefficients + b_val
    return Z


def clean_mask(mask):
    # 图中只有一条连通河面，去除孤立白斑并填补小孔。
    components, count = label(mask, structure=np.ones((3, 3)))
    if count == 0:
        return mask
    areas = np.bincount(components.ravel())
    river = components == (np.argmax(areas[1:]) + 1)

    background, _ = label(~river)
    holes = np.bincount(background.ravel()) <= 24
    holes[0] = False
    border = np.r_[background[0], background[-1], background[:, 0], background[:, -1]]
    holes[border] = False
    return river | holes[background]


def holdout_points(annotated_xy):
    with np.load(HOLDOUT_FILE) as data:
        xy = data['xy']
        water = data['tag'] == 'B'
    # 11 × 11 邻域的半径是 5，排除与训练标注邻域重叠的评估点。
    distance, _ = cKDTree(annotated_xy).query(xy, p=np.inf)
    keep = distance > 10
    return xy[keep], water[keep], len(xy)


def holdout_scores(mask, xy, water):
    predicted = mask[xy[:, 1], xy[:, 0]]
    correct = np.count_nonzero(predicted == water)
    intersection = np.count_nonzero(predicted & water)
    union = np.count_nonzero(predicted | water)
    accuracy = f'{correct / len(water):.2%}' if len(water) else '—'
    iou = f'{intersection / union:.2%}' if union else '—'
    return accuracy, iou


def experiment_report(x_points, y_points, annotated_xy, timings, models, raw_masks, clean_masks):
    test_xy, water, total = holdout_points(annotated_xy)
    n_water = np.count_nonzero(water)
    names = {'qp': '线性原始', 'dual': '线性对偶', 'rbf': 'RBF 对偶'}
    support = {
        'qp': '—',
        'dual': np.count_nonzero(np.abs(models['dual'][2]) > 1e-8),
        'rbf': len(models['rbf'][0]),
    }

    lines = [
        '# SVM 水面分割实验比较',
        '',
        f'三种方法共用 {len(x_points)} 个训练像素、{x_points.shape[1]} 个特征和 C = {C0:g}。'
        f'RBF 核取 γ = {GAMMA:g}。',
        f'评估读取 [留出标注](../data/{HOLDOUT_FILE.name}) 的 {total} 个点，'
        f'排除与训练标注的 11×11 特征邻域重叠的 {total - len(test_xy)} 个点后，'
        f'保留 {len(test_xy)} 个，其中水面 {n_water} 个，非水面 {len(water) - n_water} 个。'
        '评估仅反映本图的留出点表现。',
        '',
        '| 方法 | 训练（秒） | 整图预测（秒） | 支持向量 | 原始准确率 | 原始水面 IoU | 清理后准确率 | 清理后水面 IoU |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for method in METHODS:
        train_time, predict_time = timings[method]
        raw_acc, raw_iou = holdout_scores(raw_masks[method], test_xy, water)
        clean_acc, clean_iou = holdout_scores(clean_masks[method], test_xy, water)
        lines.append(
            f'| {names[method]} | {train_time:.4f} | {predict_time:.4f} | {support[method]} | '
            f'{raw_acc} | {raw_iou} | {clean_acc} | {clean_iou} |'
        )

    w, b = models['qp']
    w_dual, b_dual, beta = models['dual']
    slack = np.maximum(0, 1 - y_points * (x_points @ w + b))
    primal = 0.5 * (w @ w) + C0 * slack.sum()
    dual = beta @ y_points - 0.5 * (w_dual @ w_dual)
    lines += [
        '',
        '计时取自单次运行。训练计时包括矩阵构造和求解。训练及预测均不含特征提取、'
        '后处理、绘图和文件写入。水面 IoU 为预测与标注水面的交集除以并集。',
        '',
        f'线性原始目标值为 {primal:.8f}，对偶目标值为 {dual:.8f}，绝对差为 {abs(primal - dual):.3g}。'
        '目标值用标准软间隔公式重算，省去求解时用于保证矩阵正定的微小正则项。',
        f'两种线性解的权重差范数为 {np.linalg.norm(w - w_dual):.3g}，'
        f'偏置差的绝对值为 {abs(b - b_dual):.3g}。',
        '',
        '| 整图比较 | 原始掩膜差异 | 清理后掩膜差异 |',
        '| --- | ---: | ---: |',
    ]
    for left, right in (('qp', 'dual'), ('dual', 'rbf')):
        count = raw_masks[left].size
        raw_diff = np.count_nonzero(raw_masks[left] != raw_masks[right])
        clean_diff = np.count_nonzero(clean_masks[left] != clean_masks[right])
        lines.append(
            f'| {names[left]} 与 {names[right]} | {raw_diff:,} 像素（{raw_diff / count:.4%}） | '
            f'{clean_diff:,} 像素（{clean_diff / count:.4%}） |'
        )

    changed = [
        f'{names[method]} {np.count_nonzero(raw_masks[method] != clean_masks[method]):,} 像素'
        for method in METHODS
    ]
    lines += ['', '后处理改动量为' + '、'.join(changed) + '。']
    return '\n'.join(lines) + '\n'


def comparison_figure(image, raw_masks, clean_masks):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.transforms import offset_copy

    titles = {
        'qp': 'Linear SVM (primal)',
        'dual': 'Linear SVM (dual)',
        'rbf': 'RBF SVM (dual)',
    }
    columns = len(raw_masks) + 1
    margin = 0.25
    top_margin = 0.375
    col_gap, row_gap = 0.25, 0.375
    width = columns * PANEL_SIZE + (columns - 1) * col_gap + 2 * margin
    height = 2 * PANEL_SIZE + row_gap + top_margin + margin

    fig = Figure(figsize=(width, height), facecolor='white')
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(2, columns)
    fig.subplots_adjust(
        left=margin / width, right=1 - margin / width,
        bottom=margin / height, top=1 - top_margin / height,
        wspace=col_gap / PANEL_SIZE, hspace=row_gap / PANEL_SIZE,
    )

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    panels = [(fig.add_subplot(grid[:, 0]), 'Original image', rgb)]
    for row, (state, masks) in enumerate((('Raw', raw_masks), ('Cleaned', clean_masks))):
        for col, (method, mask) in enumerate(masks.items(), start=1):
            panels.append((fig.add_subplot(grid[row, col]), f'{titles[method]} · {state}', mask))

    for letter, (ax, title, panel) in zip('abcdefg', panels):
        ax.imshow(panel, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#b8b8b8')
            spine.set_linewidth(0.55)

        letter_position = offset_copy(ax.transAxes, fig=fig, y=5, units='points')
        title_position = offset_copy(ax.transAxes, fig=fig, x=13, y=5, units='points')
        ax.text(0, 1, letter, transform=letter_position, va='bottom',
                fontfamily='DejaVu Sans', fontsize=12, fontweight='bold', color='#222222')
        ax.text(0, 1, title, transform=title_position, va='bottom',
                fontfamily='DejaVu Sans', fontsize=11, color='#222222')

    return fig


def main(default_method='all'):
    parser = argparse.ArgumentParser(description='用 7 个像素特征进行 SVM 水面分割')
    parser.add_argument('--method', choices=('all', *METHODS), default=default_method)
    parser.add_argument('--points', nargs='+', type=Path, default=POINT_FILES,
                        help='指定标注 NPZ 文件，超过 1000 点的文件按网格抽样')
    parser.add_argument('--no-gui', action='store_true', help='只保存结果，不打开窗口')
    args = parser.parse_args()

    image = cv2.imread(str(IMAGE))
    train_xy, y_points, annotated_xy = read_points(args.points)
    features = pixel_features(image, train_xy)
    x_points = features[train_xy[:, 1], train_xy[:, 0]]
    img_flat = features.reshape(-1, features.shape[-1])
    print(f'训练样本：{len(x_points)} 个，{x_points.shape[1]} 维特征')

    fit = {'qp': fit_primal_qp, 'dual': fit_linear_dual, 'rbf': fit_rbf}
    methods = METHODS if args.method == 'all' else (args.method,)
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    models, timings, raw_masks, clean_masks = {}, {}, {}, {}

    for method in methods:
        start = perf_counter()
        model = fit[method](x_points, y_points)
        trained = perf_counter()

        if method == 'rbf':
            Z = predict_rbf(img_flat, *model)
        else:
            w, b_val = model[:2]
            Z = np.dot(img_flat, w) + b_val
        timings[method] = trained - start, perf_counter() - trained
        models[method] = model

        raw = Z.reshape(image.shape[:2]) >= 0
        cleaned = clean_mask(raw)
        raw_masks[method], clean_masks[method] = raw, cleaned

        for suffix, mask in (('', raw), ('_clean', cleaned)):
            black_white = mask.astype(np.uint8) * 255
            path = output / f'svm_{method}{suffix}.png'
            _, png = cv2.imencode('.png', black_white)
            path.write_bytes(png.tobytes())
            print(f'分割结果已保存到 {path}')

    if args.method == 'all':
        report = experiment_report(x_points, y_points, annotated_xy, timings, models, raw_masks, clean_masks)
        path = output / 'experiment.md'
        path.write_text(report, encoding='utf-8')
        print()
        print(report, end='')
        print(f'实验比较已保存到 {path}')

    if args.no_gui and args.method != 'all':
        return

    fig = comparison_figure(image, raw_masks, clean_masks)

    if args.method == 'all':
        path = output / 'comparison.png'
        export_dpi = int(np.ceil(max(image.shape[:2]) / PANEL_SIZE))
        fig.savefig(path, dpi=export_dpi, facecolor='white')
        print(f'对比图已保存到 {path}')

    if not args.no_gui:
        fig.set_dpi(min(132, 1760 / fig.get_figwidth(), 960 / fig.get_figheight()))
        # 预览缩小时只对原图平滑采样，掩膜保留最近邻插值。
        fig.axes[0].images[0].set_interpolation('lanczos')
        fig.canvas.draw()
        canvas = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        title = 'SVM Segmentation Comparison'
        cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(title, canvas.shape[1], canvas.shape[0])
        cv2.imshow(title, canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
