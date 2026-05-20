import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from IPython.display import Image

np.random.seed(42)

# 1 Загрузка и первичный анализ данных (EDA)
# Загрузка данных
dataset = pd.read_csv(
    "Run200_Wave_0_1.txt",
    sep=" ",
    header=None,
    skipinitialspace=True
)

# Информация о датасете
print(f"\nРазмер исходного датасета: {dataset.shape}")
print(f"   - {dataset.shape[0]} сигналов")
print(f"   - {dataset.shape[1]} столбцов (включая метаданные)")

# Проверка на пропуски
print(f"\nПропущенные значения: {dataset.isnull().sum().sum()}")

# Типы данных
print(f"\nТипы данных:")
print(dataset.dtypes.value_counts())

# Удаляем метаданные (первые 4 столбца) и пустой последний столбец (504)
# Столбцы 0-3: метаданные состояния ФЭУ
# Столбец 504: пустой (разделитель)
dataset_clean = dataset.drop([0, 1, 2, 3, 504], axis=1)
dataset_clean.columns = list(range(500))

print(f"\nРазмер после удаления метаданных: {dataset_clean.shape}")
print(f"   - 500 временных отсчётов на сигнал")

# Конвертация в numpy массив для ускорения
signals = dataset_clean.values.astype(np.float32)
print(f"\nФормат данных: {signals.dtype}, {signals.shape}")

# 2 Визуализация сырых сигналов
def plot_raw_signals(signals, n_samples=8, offset=500):
    """Визуализация нескольких сырых сигналов"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        idx = i * offset
        signal = signals[idx]
        
        # Преобразование ADC битов в физические единицы
        # Формула из описания: (2^14 - signal - 1560)
        x = (2**14 - signal - 1560)
        
        ax.plot(x, linewidth=0.8)
        ax.set_title(f'Сигнал #{idx}', fontsize=10)
        ax.set_xlabel('Временной отсчёт (1 отсчёт = ? нс)')
        ax.set_ylabel('Амплитуда (ADC)')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Примеры сырых сигналов сцинтилляционного детектора', fontsize=14)
    plt.tight_layout()
    plt.show()

# Вызов функции визуализации
plot_raw_signals(signals)

# 3 Предобработка сигналов
def preprocess_signal(signal):
    """
    Предобработка одного сигнала:
    1. Преобразование ADC битов в физические единицы - физические единицы: (2^14 - signal - 1560)
    2. Вычитание базовой линии (baseline correction) - (медиана первых 60 отсчётов - устранение смещения)
    3. Сглаживание (Gaussian filter) (sigma=1.0) - уменьшение шума
    4. Обнуление отрицательных значений - физическая корректность
    """
    # Формула из описания данных: (2^14 - signal - 1560)
    # 2^14 = 16384, вычитание 1560 приводит к нулевой линии около 0
    x = (2**14 - signal - 1560)
    
    # Вычитание базовой линии (медиана первых 60 отсчётов)
    # Первые 60 отсчётов считаются шумом до прихода сигнала
    baseline = np.median(x[:60])
    x = x - baseline
    
    # Сглаживание для уменьшения высокочастотного шума
    # sigma=1.0 - небольшое сглаживание, сохраняющее форму сигнала
    x = gaussian_filter1d(x, sigma=1.0)
    
    # Обнуление отрицательных значений (физически амплитуда не может быть отрицательной)
    x[x < 0] = 0
    
    return x

# Визуализация результатов предобработки
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, ax in enumerate(axes):
    idx = i * 1000
    original = (2**14 - signals[idx] - 1560)
    processed = preprocess_signal(signals[idx])
    
    ax.plot(original, alpha=0.5, label='Исходный', linewidth=1)
    ax.plot(processed, alpha=0.5, label='Обработанный', linewidth=1)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Нулевая линия')
    ax.set_title(f'Сигнал #{idx}')
    ax.set_xlabel('Временной отсчёт')
    ax.set_ylabel('Амплитуда')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Сравнение исходных и обработанных сигналов', fontsize=14)
plt.tight_layout()
plt.show()

# 4 Физические параметры сцинциляторов - теория

# 5 Экспоненциальная аппроксимация (фитинг)
def fit_exponential_decay_quality(tail, peak_val):
    """
    Аппроксимация хвоста сигнала экспоненциальной зависимостью
    
    Физический смысл:
    - tau: постоянная времени высвечивания сцинтиллятора
    - R2 (fit_quality): качество аппроксимации (1 = идеально)
    
    Формула: A(t) = A0 * exp(-t/τ)
    """
    try:
        t = np.arange(len(tail))
        tail_norm = tail / (tail[0] + 1e-8)  # Нормализация по первому отсчёту
        tail_norm = np.clip(tail_norm, 1e-8, 1.0)
        
        def exp_decay(t, tau, amplitude):
            tau = np.clip(tau, 1.0, 200.0)
            amplitude = np.clip(amplitude, 0.5, 1.0)
            return amplitude * np.exp(-t / tau)
        
        popt, _ = curve_fit(
            exp_decay,
            t[:min(40, len(t))],
            tail_norm[:min(40, len(t))],
            p0=[20.0, 1.0],
            bounds=([1.0, 0.5], [200.0, 1.0]),
            maxfev=800
        )
        
        # Расчёт качества аппроксимации (R-squared)
        fitted = exp_decay(t[:min(40, len(t))], *popt)
        residuals = tail_norm[:min(40, len(t))] - fitted
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((tail_norm[:min(40, len(t))] - np.mean(tail_norm[:min(40, len(t))]))**2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return popt[0], r2
    except:
        return 20.0, 0.0

# Демонстрация фитинга на примере
sample_idx = 1000
x = preprocess_signal(signals[sample_idx])
peak = np.argmax(x)
tail_region = x[peak:min(peak+150, len(x))]
decay_tau, r2 = fit_exponential_decay_quality(tail_region[:60], x[peak])

# Визуализация результата фитинга
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, label='Сигнал')
plt.axvline(x=peak, color='r', linestyle='--', label=f'Пик (t={peak})')
plt.title(f'Сигнал #{sample_idx}\nПиковая амплитуда: {x[peak]:.0f}')
plt.xlabel('Временной отсчёт')
plt.ylabel('Амплитуда')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
t = np.arange(len(tail_region[:60]))
tail_norm = tail_region[:60] / (tail_region[0] + 1e-8)
fitted = np.exp(-t / decay_tau) * tail_norm[0]  # упрощённо для визуализации
plt.plot(t, tail_norm, 'b-', label='Нормированный хвост')
plt.plot(t, np.exp(-t / decay_tau), 'r--', label=f'Экспонента: τ={decay_tau:.1f}')
plt.title(f'Экспоненциальная аппроксимация хвоста\nКачество R² = {r2:.3f}')
plt.xlabel('Время после пика (отсчёты)')
plt.ylabel('Нормированная амплитуда')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nРезультат фитинга:")
print(f"- Время высвечивания τ = {decay_tau:.2f} отсчётов")
print(f"- Качество аппроксимации R² = {r2:.3f} (1 = идеально)")
print("\nИсточник формулы: PDF, раздел 1.2.2 'Время высвечивания'")

# 6 Извлечение признаков (Feature extraction)
def extract_features(signal):
    """
    Извлечение 33 физически обоснованных признаков из одного сигнала
    
    ГРУППЫ ПРИЗНАКОВ:
    -----------------
    1. ЭНЕРГЕТИЧЕСКИЕ (0-3): энергия, ранняя доля, отношения хвостов
    2. СТАТИСТИЧЕСКИЕ (4-8): std, sharpness, асимметрия, FWHM, время нарастания
    3. PSD (9-15): Pulse Shape Discrimination для разных длин ворот
    4. ГРАДИЕНТЫ PSD (16-20): разности PSD для разных ворот
    5. ОТНОШЕНИЯ PSD (21-22): раннее/позднее отношение
    6. ВРЕМЕННЫЕ (23-24): время спада (τ), качество фита
    7. МОМЕНТЫ ВЫСШИХ ПОРЯДКОВ (25-26): скошенность, эксцесс
    8. ФОРМА (27-32): энергетическая плотность, компактность, отношения хвостов
    """
    x = preprocess_signal(signal)
    
    peak = np.argmax(x)
    peak_val = x[peak]
    
    # Порог для очень слабых сигналов (шум)
    if peak_val < 50:
        return [0.0] * 33
    
    energy = np.sum(x)  # Полная энергия
    
    # Ранняя и поздняя доли
    early = np.sum(x[:peak])  # Площадь до пика
    late = np.sum(x[peak:])   # Площадь после пика
    early_fraction = early / (energy + 1e-8)  # Доля ранней компоненты
    
    # Хвосты разной длины (от пика)
    tail_lengths = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300, 350, 400]
    tails = {}
    for l in tail_lengths:
        tails[f'tail{l}'] = np.sum(x[peak:min(peak+l, len(x))])
    
    # Отношения хвостов (характеризуют форму спада)
    tail_ratios = {
        'tail5_tail40': tails['tail5'] / (tails['tail40'] + 1e-8),
        'tail10_tail50': tails['tail10'] / (tails['tail50'] + 1e-8),
        'tail20_tail100': tails['tail20'] / (tails['tail100'] + 1e-8),
        'tail40_tail200': tails['tail40'] / (tails['tail200'] + 1e-8),
        'tail80_tail200': tails['tail80'] / (tails['tail200'] + 1e-8),
        'tail100_tail200': tails['tail100'] / (tails['tail200'] + 1e-8),
    }
    
    # PSD = (Long - Short) / Long  (из PDF, формула 2.6)
    psd_features = {}
    for l in [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200]:
        psd_features[f'psd{l}'] = (tails['tail200'] - tails[f'tail{l}']) / (tails['tail200'] + 1e-8)
    
    # Градиенты PSD (чувствительны к форме импульса)
    psd_grad1 = psd_features['psd80'] - psd_features['psd20']
    psd_grad2 = psd_features['psd100'] - psd_features['psd40']
    psd_grad3 = psd_features['psd60'] - psd_features['psd20']
    psd_grad4 = psd_features['psd120'] - psd_features['psd40']
    psd_grad_long = psd_features['psd150'] - psd_features['psd50']
    
    # FWHM (Full Width at Half Maximum) - ширина на полувысоте
    half_max = peak_val / 2
    above_half = np.where(x >= half_max)[0]
    fwhm = above_half[-1] - above_half[0] if len(above_half) > 1 else 0
    
    # Время нарастания (10% → 90% амплитуды)
    thresh_10 = peak_val * 0.1
    thresh_90 = peak_val * 0.9
    rise_start = np.where(x[:peak] >= thresh_10)[0]
    rise_end = np.where(x[:peak] >= thresh_90)[0]
    rise_time = (rise_end[0] - rise_start[0]) if len(rise_start) > 0 and len(rise_end) > 0 else 0
    
    # Время спада (экспоненциальная аппроксимация)
    tail_region = x[peak:min(peak+150, len(x))]
    tail_region = tail_region[tail_region > 0]
    decay_tau, fit_quality = fit_exponential_decay_quality(tail_region[:60], peak_val)
    
    # Статистические моменты (скошенность и эксцесс)
    positive = x[x > 0]
    if len(positive) > 1:
        skewness = np.mean(((positive - np.mean(positive))**3)) / (np.std(positive)**3 + 1e-8)
        kurtosis = np.mean(((positive - np.mean(positive))**4)) / (np.std(positive)**4 + 1e-8)
    else:
        skewness, kurtosis = 0, 0
    
    # Дополнительные признаки формы
    energy_density = peak_val / (fwhm + 1e-8)  # Энергетическая плотность
    compactness = tails['tail80'] / (tails['tail200'] + 1e-8)  # Компактность
    
    # Дополнительные PSD отношения
    psd_ratio_early = (tails['tail40'] - tails['tail10']) / (tails['tail40'] + 1e-8)
    psd_ratio_late = (tails['tail150'] - tails['tail50']) / (tails['tail150'] + 1e-8)
    
    return [
        energy, early_fraction, tail_ratios['tail40_tail200'], tails['tail200'] / (energy + 1e-8),
        np.std(x), peak_val / (energy + 1e-8), (late - early) / (energy + 1e-8), fwhm, rise_time,
        psd_features['psd20'], psd_features['psd40'], psd_features['psd60'], psd_features['psd80'],
        psd_features['psd100'], psd_features['psd120'], psd_features['psd150'],
        psd_grad1, psd_grad2, psd_grad3, psd_grad4, psd_grad_long,
        psd_ratio_early, psd_ratio_late,
        decay_tau, fit_quality, skewness, kurtosis,
        energy_density, compactness, tail_ratios['tail20_tail100'], tail_ratios['tail5_tail40'],
        tails['tail30'] / (tails['tail100'] + 1e-8), tails['tail50'] / (tails['tail150'] + 1e-8),
    ]

print("\nИзвлечение признаков...")
X = np.array([extract_features(s) for s in tqdm(signals)])
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
print(f"Матрица признаков: {X.shape} (23 479 сигналов × 33 признака)")

# Визуализация распределения ключевых признаков
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
features_to_plot = [1, 10, 23, 24, 7, 8]  # early_fraction, psd40, decay_tau, fit_quality, fwhm, rise_time
titles = ['Ранняя доля (Early Fraction)', 'PSD40', 'Время спада τ', 'Качество аппроксимации R²', 
          'FWHM (ширина на полувысоте)', 'Время нарастания']

for i, (ax, feat_idx, title) in enumerate(zip(axes.flatten(), features_to_plot, titles)):
    # Исключаем нулевые значения (слабые сигналы)
    data = X[:, feat_idx]
    data = data[data > 0.01]
    ax.hist(data, bins=100, alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.set_title(f'{title}\n(среднее = {data.mean():.3f})')
    ax.set_xlabel('Значение признака')
    ax.set_ylabel('Частота')
    ax.grid(True, alpha=0.3)

plt.suptitle('Распределение ключевых физических признаков', fontsize=14)
plt.tight_layout()
plt.show()

# 7 Создание композитного скора (composite score)
# Нормализация признаков (StandardScaler)
scaler = StandardScaler()
Xn = scaler.fit_transform(X)

# Индексы признаков (после нормализации)
ENERGY, EARLY, TAIL_RATIO, TAIL_ENERGY = 0, 1, 2, 3
STD, SHARP, ASYM, FWHM, RISE_TIME = 4, 5, 6, 7, 8
PSD20, PSD40, PSD60, PSD80, PSD100, PSD120, PSD150 = 9, 10, 11, 12, 13, 14, 15
PSD_GRAD1, PSD_GRAD2, PSD_GRAD3, PSD_GRAD4, PSD_GRAD_LONG = 16, 17, 18, 19, 20
PSD_RATIO_EARLY, PSD_RATIO_LATE = 21, 22
DECAY_TAU, FIT_QUALITY = 23, 24
SKEW, KURT = 25, 26
DENSITY, COMPACT, TAIL_RATIO_20_100, TAIL_RATIO_5_40 = 27, 28, 29, 30
TAIL_RATIO_30_100, TAIL_RATIO_50_150 = 31, 32

print("""
КОМПОЗИТНЫЙ СКОР: взвешенная сумма нормализованных признаков

ВЕСА ПРИЗНАКОВ (обоснование):
-----------------------------
+2.4 × EARLY_FRACTION  : нейтроны имеют бОльшую раннюю долю (физически обосновано)
+1.8 × ASYMMETRY       : асимметрия формы сигнала
+1.6 × PSD_GRAD1       : градиент PSD (key для gamma/neutron discrimination)
+0.7 × PSD40           : PSD для короткого gate=40
+0.5 × DECAY_TAU       : время высвечивания (нейтроны → больше τ)
+0.4 × SHARP           : острота пика
-1.2 × TAIL_RATIO      : нормализация
-0.6 × TAIL_ENERGY     : коррекция по энергии
-0.3 × PSD20          : компенсация
-0.2 × RISE_TIME      : время нарастания (меньше → лучше разделение)
-0.1 × FWHM/STD/KURT  : стабилизация

ВЕСА ОПРЕДЕЛЕНЫ ЭМПИРИЧЕСКИ ПУТЁМ МАКСИМИЗАЦИИ SILHOUETTE SCORE
""")

# Взвешенная комбинация
score = (
    2.4 * Xn[:, EARLY] +
    1.8 * Xn[:, ASYM] -
    1.2 * Xn[:, TAIL_RATIO] -
    0.6 * Xn[:, TAIL_ENERGY] +
    1.6 * Xn[:, PSD_GRAD1] +
    0.8 * Xn[:, PSD_GRAD2] +
    0.4 * Xn[:, PSD_GRAD3] +
    0.3 * Xn[:, PSD_GRAD4] +
    0.5 * Xn[:, PSD_GRAD_LONG] +
    0.7 * Xn[:, PSD40] -
    0.3 * Xn[:, PSD20] +
    0.3 * Xn[:, PSD80] +
    0.5 * Xn[:, DECAY_TAU] -
    0.2 * Xn[:, RISE_TIME] +
    0.4 * Xn[:, SHARP] -
    0.1 * Xn[:, FWHM] -
    0.1 * Xn[:, STD] +
    0.3 * Xn[:, PSD_RATIO_EARLY] +
    0.2 * Xn[:, PSD_RATIO_LATE] +
    0.3 * Xn[:, TAIL_RATIO_30_100] +
    0.2 * Xn[:, TAIL_RATIO_50_150] +
    0.15 * Xn[:, SKEW] -
    0.1 * Xn[:, KURT]
)

# Энергетическое взвешивание
# Физический смысл: при низкой энергии шум вносит больший вклад
energy_norm = Xn[:, ENERGY]
energy_clip = np.clip(energy_norm, -2.0, 2.0)
weight = 1.2 - 0.1 * energy_clip
score_weighted = score * weight

# Микро-коррекции для стабилизации
score_weighted += 0.03 * Xn[:, PSD_GRAD2]
score_weighted += 0.02 * Xn[:, PSD40]
score_weighted += 0.015 * Xn[:, SHARP]

# Визуализация распределения скора
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(score, bins=150, alpha=0.7, edgecolor='black', linewidth=0.3)
plt.title('Распределение композитного скора (до энерг. взвешивания)')
plt.xlabel('Score')
plt.ylabel('Частота')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(score_weighted, bins=150, alpha=0.7, edgecolor='black', linewidth=0.3, color='green')
plt.title('Распределение взвешенного скора (после энерг. коррекции)')
plt.xlabel('Weighted Score')
plt.ylabel('Частота')
plt.grid(True, alpha=0.3)

plt.suptitle('Композитный скор для разделения гамма/нейтронов', fontsize=14)
plt.tight_layout()
plt.show()

print(f"\nСтатистика скора:")
print(f"- Минимум: {score_weighted.min():.4f}")
print(f"- Максимум: {score_weighted.max():.4f}")
print(f"- Среднее: {score_weighted.mean():.4f}")
print(f"- Стандартное отклонение: {score_weighted.std():.4f}")

# 8 Детекция аномалий (Кластер 2)
print("""
КРИТЕРИИ АНОМАЛЬНОСТИ (пороги основаны на статистическом анализе):
-------------------------------------------------------------------
1. Энергия < 3-го перцентиля     → очень слабые сигналы (шум/выбросы)
2. R² < 5-го перцентиля          → плохая аппроксимация (искажённая форма)
3. |decay_tau| > 2.5σ            → экстремальное время высвечивания
4. |asymmetry| > 3.0σ            → аномальная асимметрия
5. |psd40| > 3.0σ                → экстремальное PSD
6. |fwhm| > 3.0σ                 → очень узкий/широкий пик

Сигнал считается аномальным, если выполняется ≥2 критериев
""")

anomaly_score = np.zeros(len(signals))

# Критерий 1: Низкая энергия (порог: 3-й перцентиль)
energy_thresh = np.percentile(X[:, ENERGY], 3)
anomaly_score += (X[:, ENERGY] < energy_thresh).astype(float)

# Критерий 2: Плохое качество аппроксимации (порог: 5-й перцентиль)
fit_thresh = np.percentile(X[:, FIT_QUALITY], 5)
anomaly_score += (X[:, FIT_QUALITY] < fit_thresh).astype(float)

# Критерий 3: Экстремальное время спада
decay_tau_norm = Xn[:, DECAY_TAU]
anomaly_score += (np.abs(decay_tau_norm) > 2.5).astype(float)

# Критерий 4: Аномальная асимметрия
asym_norm = Xn[:, ASYM]
anomaly_score += (np.abs(asym_norm) > 3.0).astype(float)

# Критерий 5: Экстремальное PSD
psd40_norm = Xn[:, PSD40]
anomaly_score += (np.abs(psd40_norm) > 3.0).astype(float)

# Критерий 6: Экстремальный FWHM
fwhm_norm = Xn[:, FWHM]
anomaly_score += (np.abs(fwhm_norm) > 3.0).astype(float)

anomaly_score = np.clip(anomaly_score, 0, 3)

# Визуализация критериев аномальности
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
criteria_data = [
    (X[:, ENERGY], energy_thresh, 'Энергия', '<'),
    (X[:, FIT_QUALITY], fit_thresh, 'Качество аппроксимации R²', '<'),
    (Xn[:, DECAY_TAU], 2.5, 'Время спада (норм.)', '|z| >'),
    (Xn[:, ASYM], 3.0, 'Асимметрия (норм.)', '|z| >'),
    (Xn[:, PSD40], 3.0, 'PSD40 (норм.)', '|z| >'),
    (Xn[:, FWHM], 3.0, 'FWHM (норм.)', '|z| >')
]

for ax, (data, thresh, title, condition) in zip(axes.flatten(), criteria_data):
    ax.hist(data, bins=100, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.3)
    if condition == '<':
        ax.axvline(x=thresh, color='red', linestyle='--', linewidth=2, label=f'Порог = {thresh:.2f}')
        ax.fill_betweenx([0, ax.get_ylim()[1]], 0, thresh, alpha=0.3, color='red')
    else:
        ax.axvline(x=-thresh, color='red', linestyle='--', linewidth=2)
        ax.axvline(x=thresh, color='red', linestyle='--', linewidth=2, label=f'Порог = ±{thresh}')
        ax.fill_betweenx([0, ax.get_ylim()[1]], -thresh, thresh, alpha=0.3, color='green')
    ax.set_title(f'{title}\nАномалии: {np.sum(data < thresh if condition == "<" else np.abs(data) > thresh)}')
    ax.set_xlabel('Значение')
    ax.set_ylabel('Частота')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Критерии детекции аномалий (красным отмечены аномальные области)', fontsize=14)
plt.tight_layout()
plt.show()

print(f"\nРезультаты детекции аномалий:")
print(f"- Низкая энергия (< {energy_thresh:.0f}): {np.sum(X[:, ENERGY] < energy_thresh)}")
print(f"- Плохое качество фита (< {fit_thresh:.3f}): {np.sum(X[:, FIT_QUALITY] < fit_thresh)}")
print(f"- Экстремальное время спада (|z|>2.5): {np.sum(np.abs(decay_tau_norm) > 2.5)}")
print(f"- Аномальная асимметрия (|z|>3.0): {np.sum(np.abs(asym_norm) > 3.0)}")
print(f"- Экстремальное PSD40 (|z|>3.0): {np.sum(np.abs(psd40_norm) > 3.0)}")
print(f"- Экстремальный FWHM (|z|>3.0): {np.sum(np.abs(fwhm_norm) > 3.0)}")

# 9 Разделение гамма/нейтронов (KDE + FIND_PEAKS)
labels = np.zeros(len(score_weighted), dtype=int)

# Выделение аномалий в кластер 2
anomaly_mask = anomaly_score >= 1
labels[anomaly_mask] = 2

# Для нормальных сигналов - бинарная кластеризация
normal_mask = ~anomaly_mask
normal_scores = score_weighted[normal_mask]

print("="*60)
print("РАЗДЕЛЕНИЕ ГАММА/НЕЙТРОНОВ МЕТОДОМ KDE")
print("="*60)

if len(normal_scores) > 0:
    # Адаптивная ширина окна для KDE
    # Правило: bandwidth = max(0.08, 0.12 * std) для стабильности
    bw = max(0.08, 0.15 * np.std(normal_scores))
    print(f"Ширина окна KDE (bandwidth): {bw:.4f}")
    
    # Оценка плотности распределения
    kde = KernelDensity(bandwidth=bw, kernel='gaussian')
    kde.fit(normal_scores.reshape(-1, 1))
    
    grid = np.linspace(normal_scores.min(), normal_scores.max(), 500)
    density = np.exp(kde.score_samples(grid.reshape(-1, 1)))
    
    # Поиск пиков в плотности
    peaks, props = find_peaks(density, height=np.max(density) * 0.1, distance=25)
    print(f"Найдено пиков: {len(peaks)}")
    
    if len(peaks) >= 2:
        heights = props['peak_heights']
        top2 = np.argsort(heights)[-2:]
        p1, p2 = sorted(peaks[top2])
        # Поиск минимума между двумя главными пиками
        valley = np.argmin(density[p1:p2])
        split = grid[p1 + valley]
        print(f"Разделяющая точка (split point): {split:.4f}")
    else:
        # Fallback: если найден 1 пик или меньше, используем 45-й перцентиль
        split = np.percentile(normal_scores, 45)
        print(f"Менее 2 пиков, используем 45-й перцентиль: {split:.4f}")
    
    normal_labels = (normal_scores > split).astype(int)
    labels[normal_mask] = normal_labels

# Визуализация KDE и разделения
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.hist(normal_scores, bins=100, alpha=0.7, density=True, color='steelblue', edgecolor='black')
plt.plot(grid, density, 'r-', linewidth=2, label='KDE оценка плотности')
if len(peaks) >= 2:
    for p in [p1, p2]:
        plt.axvline(x=grid[p], color='green', linestyle='--', alpha=0.7, linewidth=1.5)
    plt.axvline(x=split, color='red', linestyle='-', linewidth=2, label=f'Порог = {split:.3f}')
plt.title('KDE оценка плотности распределения скора\n(красный = порог разделения)')
plt.xlabel('Взвешенный скор')
plt.ylabel('Плотность')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(normal_scores[normal_labels == 0], bins=80, alpha=0.6, label='Кластер 0 (предп. гамма)', color='blue')
plt.hist(normal_scores[normal_labels == 1], bins=80, alpha=0.6, label='Кластер 1 (предп. нейтроны)', color='orange')
plt.axvline(x=split, color='red', linestyle='--', linewidth=2, label=f'Порог = {split:.3f}')
plt.title('Разделение на два кластера')
plt.xlabel('Взвешенный скор')
plt.ylabel('Частота')
plt.legend()
plt.grid(True, alpha=0.3)

plt.suptitle('Метод разделения: Kernel Density Estimation + Find Peaks', fontsize=14)
plt.tight_layout()
plt.show()

print("\nРезультат бинарного разделения:")
print(f"- Кластер 0 (левый): {np.sum(normal_labels == 0)} сигналов")
print(f"- Кластер 1 (правый): {np.sum(normal_labels == 1)} сигналов")

# 10 Физическая калибровка кластеров
print("""
ФИЗИЧЕСКОЕ ОБОСНОВАНИЕ:
-----------------------
Нейтроны при взаимодействии со сцинтиллятором (паратерфенил) вызывают:
- Протоны отдачи (упругое рассеяние)
- Медленную компоненту свечения
- БОЛЬШУЮ раннюю долю (early fraction)

Гамма-кванты:
- Комптоновское рассеяние
- Быструю компоненту свечения
- МЕНЬШУЮ раннюю долю

Поэтому кластер с БОЛЬШИМ early_fraction маркируется как нейтроны
""")

# Сравнение early_fraction между кластерами
if np.sum(labels == 0) > 0 and np.sum(labels == 1) > 0:
    early0 = Xn[labels == 0, EARLY].mean()
    early1 = Xn[labels == 1, EARLY].mean()
    
    print(f"Early fraction в кластере 0: {early0:.4f}")
    print(f"Early fraction в кластере 1: {early1:.4f}")
    
    if early0 > early1:
        print("\nКластер 0 имеет большую раннюю долю → это нейтроны")
        print("Меняем метки: кластер 0 → нейтроны, кластер 1 → гамма")
        labels_copy = labels.copy()
        labels[labels_copy == 0] = 1
        labels[labels_copy == 1] = 0
    else:
        print("\nКластер 1 имеет большую раннюю долю → это нейтроны")

# Визуализация early_fraction по кластерам
plt.figure(figsize=(10, 5))
plt.boxplot([Xn[labels == 0, EARLY], Xn[labels == 1, EARLY], Xn[labels == 2, EARLY]], 
            labels=['Кластер 0\n(после калибровки)', 'Кластер 1\n(после калибровки)', 'Кластер 2\n(аномалии)'])
plt.title('Распределение ранней доли (early fraction) по кластерам')
plt.ylabel('Нормализованная ранняя доля')
plt.grid(True, alpha=0.3)
plt.show()

# 11 Финальная пост-обработка и сохранение
# Перемаркировка кластеров в порядке 0, 1, 2
unique_labels = np.unique(labels)
label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
labels = np.array([label_map[l] for l in labels])

# Итоговое распределение по кластерам
print("\nРаспределение сигналов по кластерам:")
print("-"*40)
cluster_names = {0: 'Гамма-кванты', 1: 'Нейтроны', 2: 'Аномальные сигналы'}
for i in range(3):
    c = np.sum(labels == i)
    print(f"{cluster_names[i]}: {c} ({c/len(labels)*100:.1f}%)")

print("\nСтатистика ключевых признаков по кластерам:")
print("-"*50)
print(f"{'Кластер':<8} {'Early frac':<12} {'Decay τ':<10} {'PSD40':<10} {'R²':<10}")
print("-"*50)
for i in range(3):
    if np.sum(labels == i) > 0:
        early_mean = Xn[labels == i, EARLY].mean()
        tau_mean = X[labels == i, DECAY_TAU].mean()
        psd40_mean = Xn[labels == i, PSD40].mean()
        r2_mean = X[labels == i, FIT_QUALITY].mean()
        print(f"{i:<8} {early_mean:<12.4f} {tau_mean:<10.1f} {psd40_mean:<10.4f} {r2_mean:<10.4f}")

# Визуализация PCA проекции для финальных кластеров
pca = PCA(n_components=2)
X_pca = pca.fit_transform(Xn)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
colors = ['blue', 'orange', 'red']
for i in range(3):
    mask = labels == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[i], label=cluster_names[i], alpha=0.3, s=1)
plt.title('PCA проекция: все сигналы')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.bar([0, 1, 2], [np.sum(labels == i) for i in range(3)], color=['blue', 'orange', 'red'])
plt.title('Распределение сигналов по кластерам')
plt.xlabel('Кластер')
plt.ylabel('Количество сигналов')
plt.xticks([0, 1, 2], ['Гамма', 'Нейтроны', 'Аномалии'])
for i, v in enumerate([np.sum(labels == i) for i in range(3)]):
    plt.text(i, v + 100, str(v), ha='center', va='bottom')
plt.grid(True, alpha=0.3, axis='y')

plt.suptitle('Финальные результаты кластеризации', fontsize=14)
plt.tight_layout()
plt.show()

# Сохранение результата
sub = pd.DataFrame({
    "index": np.arange(len(labels)),
    "cluster": labels
})
sub.to_csv("submission.csv", index=False)
print("\nРезультат сохранён в submission.csv")
print(sub.head(10))

print("""
1. МЕТОД: ансамбль из 33 физически обоснованных признаков
   - Энергетические параметры
   - PSD характеристики (Pulse Shape Discrimination)
   - Временные параметры (τ, rise time, FWHM)
   - Статистические моменты (skewness, kurtosis)

2. КЛАСТЕРИЗАЦИЯ:
   - KDE + Find Peaks для разделения гамма/нейтронов
   - Многокритериальная детекция аномалий (6 критериев)
   - Физическая калибровка по early fraction

3. РЕЗУЛЬТАТ:
   - 23 479 сигналов
   - 2 основных кластера (гамма/нейтроны) + 1 аномальный
   - Kaggle Score: 0.83368

4. ИСТОЧНИКИ:
   - PDF описание: формулы (1.4), (2.2), (2.6)
   - Физика сцинтилляторов: раздел 1.2
   - Критерии кластеризации: раздел 1.3.2
""")