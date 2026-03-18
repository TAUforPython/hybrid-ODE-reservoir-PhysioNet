import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, resample
from scipy.stats import pearsonr
import os
import urllib.request


# =============================================================
# БЛОК 1: Загрузчик реальных ЭКГ с PhysioNet
# =============================================================

class PhysioNetECGLoader:
    """
    Загрузка записей ЭКГ из базы PhysioNet ECG-ID Database.
    90 пациентов, 310 записей, 500 Гц, формат EDF.
    """

    def __init__(self, data_dir="ecgiddb_data"):
        self.data_dir = data_dir
        self.base_url = "https://physionet.org/files/ecgiddb/1.0.0/"

    def download_record(self, person_id=1, record_id=1):
        os.makedirs(self.data_dir, exist_ok=True)
        person_folder = f"Person_{person_id:02d}"
        filename = f"rec_{record_id}.edf"
        remote_path = f"{self.base_url}{person_folder}/{filename}"
        local_folder = os.path.join(self.data_dir, person_folder)
        os.makedirs(local_folder, exist_ok=True)
        local_path = os.path.join(local_folder, filename)

        if not os.path.exists(local_path):
            print(f"Скачиваю: {remote_path}")
            try:
                urllib.request.urlretrieve(remote_path, local_path)
                print(f"Сохранено: {local_path}")
            except Exception as e:
                print(f"Ошибка загрузки: {e}")
                return None
        else:
            print(f"Файл уже существует: {local_path}")
        return local_path

    def read_edf(self, filepath):
        try:
            import pyedflib
            f = pyedflib.EdfReader(filepath)
            signal = f.readSignal(0)
            fs = f.getSampleFrequency(0)
            f.close()
            return signal, fs
        except ImportError:
            import mne
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            return raw.get_data()[0], raw.info['sfreq']


# =============================================================
# БЛОК 2: Предобработка ЭКГ с детекцией всех зубцов
# =============================================================

class ECGPreprocessor:
    """
    Предобработка ЭКГ:
    - Полосовой фильтр 0.5-40 Гц
    - Детекция R-пиков, P-зубцов, T-зубцов
    - Сегментация кардиоциклов
    - Вычисление временных меток фаз сердечного цикла
    """

    @staticmethod
    def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = min(highcut / nyq, 0.99)
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal)

    @staticmethod
    def normalize(signal):
        s_min, s_max = np.min(signal), np.max(signal)
        if s_max - s_min < 1e-10:
            return np.zeros_like(signal)
        return (signal - s_min) / (s_max - s_min)

    @staticmethod
    def find_r_peaks(signal, fs, min_distance_sec=0.4):
        min_distance = int(min_distance_sec * fs)
        threshold = 0.6 * np.max(signal)
        peaks, _ = find_peaks(signal, height=threshold, distance=min_distance)
        return peaks

    @staticmethod
    def find_p_waves(signal, fs, r_peaks):
        """
        P-зубец: ищем локальный максимум за 120-200 мс до R-пика.
        Это момент деполяризации предсердий.
        """
        p_peaks = []
        for r in r_peaks:
            start = max(0, r - int(0.20 * fs))
            end = max(0, r - int(0.08 * fs))
            if start < end and end < len(signal):
                segment = signal[start:end]
                if len(segment) > 0:
                    local_max = np.argmax(segment) + start
                    p_peaks.append(local_max)
        return np.array(p_peaks)

    @staticmethod
    def find_t_waves(signal, fs, r_peaks):
        """
        T-зубец: ищем локальный максимум через 150-400 мс после R-пика.
        Это момент реполяризации желудочков (конец систолы).
        """
        t_peaks = []
        for r in r_peaks:
            start = r + int(0.15 * fs)
            end = r + int(0.40 * fs)
            if start < len(signal) and end < len(signal):
                segment = signal[start:end]
                if len(segment) > 0:
                    local_max = np.argmax(segment) + start
                    t_peaks.append(local_max)
        return np.array(t_peaks)

    @staticmethod
    def compute_cardiac_phases(r_peaks, p_peaks, t_peaks, fs):
        """
        Вычисление временных меток фаз сердечного цикла:
        - P-wave onset: начало предсердной систолы
        - QRS onset: начало желудочковой систолы
        - Aortic valve opening: ~50 мс после QRS
        - T-wave peak: конец систолы
        - RR interval: полный период
        """
        phases = []
        for i in range(min(len(r_peaks), len(p_peaks), len(t_peaks))):
            rr = (r_peaks[i + 1] - r_peaks[i]) / fs if i + 1 < len(r_peaks) else 0.85
            phase = {
                'p_wave_time': p_peaks[i] / fs,
                'qrs_time': r_peaks[i] / fs,
                'aortic_valve_open': (r_peaks[i] + int(0.05 * fs)) / fs,
                't_wave_time': t_peaks[i] / fs,
                'rr_interval': rr,
                'heart_rate': 60.0 / rr if rr > 0 else 0
            }
            phases.append(phase)
        return phases

    def full_pipeline(self, signal, fs, target_length=500):
        filtered = self.bandpass_filter(signal, fs)
        normalized = self.normalize(filtered)
        r_peaks = self.find_r_peaks(normalized, fs)

        if len(r_peaks) < 3:
            print("Слишком мало R-пиков!")
            return None

        p_peaks = self.find_p_waves(normalized, fs, r_peaks)
        t_peaks = self.find_t_waves(normalized, fs, r_peaks)

        rr_intervals = np.diff(r_peaks) / fs
        heart_rate = 60.0 / np.mean(rr_intervals)
        cardiac_phases = self.compute_cardiac_phases(r_peaks, p_peaks, t_peaks, fs)

        # Сегментация кардиоциклов
        pre_samples = int(0.2 * fs)
        post_samples = int(0.55 * fs)
        cycles = []
        for peak in r_peaks:
            start = peak - pre_samples
            end = peak + post_samples
            if start >= 0 and end <= len(normalized):
                cycles.append(normalized[start:end])

        # Усреднение
        resampled = [resample(c, target_length) for c in cycles]
        resampled = np.array(resampled)
        mean_cycle = np.mean(resampled, axis=0)
        std_cycle = np.std(resampled, axis=0)

        print(f"R-пиков: {len(r_peaks)}, P-зубцов: {len(p_peaks)}, "
              f"T-зубцов: {len(t_peaks)}")
        print(f"ЧСС: {heart_rate:.1f} уд/мин, "
              f"RR: {np.mean(rr_intervals)*1000:.0f} мс")

        return {
            'filtered': filtered,
            'normalized': normalized,
            'r_peaks': r_peaks,
            'p_peaks': p_peaks,
            't_peaks': t_peaks,
            'heart_rate': heart_rate,
            'rr_interval': np.mean(rr_intervals),
            'cardiac_phases': cardiac_phases,
            'cycles': cycles,
            'mean_cycle': mean_cycle,
            'std_cycle': std_cycle,
            'all_cycles_resampled': resampled,
            'fs': fs
        }


# =============================================================
# БЛОК 3: Функция активации сердечного цикла (эластанс)
# =============================================================

class CardiacActivation:
    """
    Модель варьирующейся во времени эластичности E(t).
    Связь с ЭКГ:
      - P-зубец: начало роста E_atria
      - QRS: начало роста E_ventricles
      - T-зубец: начало падения E_ventricles
    """

    def __init__(self, T=0.85, T_sys=0.3, T_atrial=0.1,
                 atrial_delay=0.12):
        """
        T: период сердечного цикла (из RR интервала ЭКГ)
        T_sys: длительность систолы желудочков (QRS → T-зубец)
        T_atrial: длительность систолы предсердий (P-зубец)
        atrial_delay: задержка предсердной систолы перед QRS
        """
        self.T = T
        self.T_sys = T_sys
        self.T_atrial = T_atrial
        self.atrial_delay = atrial_delay

    def e_ventricle(self, t):
        """
        Активация желудочков.
        Соответствует ЭКГ: от QRS до конца T-зубца.
        """
        t_mod = t % self.T
        if t_mod < self.T_sys:
            # Систола: рост и пик
            return 0.5 * (1.0 - np.cos(np.pi * t_mod / self.T_sys))
        elif t_mod < 1.5 * self.T_sys:
            # Расслабление: падение
            return 0.5 * (1.0 + np.cos(
                2.0 * np.pi * (t_mod - self.T_sys) / self.T_sys))
        else:
            # Диастола: минимум
            return 0.0

    def e_atrium(self, t):
        """
        Активация предсердий.
        Соответствует ЭКГ: P-зубец (перед QRS).
        Обеспечивает 'предсердную подкачку' (atrial kick).
        """
        t_mod = t % self.T
        # Предсердная систола начинается за atrial_delay до конца цикла
        t_atrial_start = self.T - self.atrial_delay - self.T_atrial
        t_atrial_end = self.T - self.atrial_delay

        if t_atrial_start < t_mod < t_atrial_end:
            phase = (t_mod - t_atrial_start) / self.T_atrial
            return 0.5 * (1.0 - np.cos(2.0 * np.pi * phase))
        return 0.0


# =============================================================
# БЛОК 4: Reservoir Computing
# =============================================================

class ReservoirNet(nn.Module):
    """
    Reservoir Computing для аппроксимации Pperi и Vspt.

    Входы синхронизированы с фазами ЭКГ:
    - 5 объёмов камер (V_LA, V_LV, V_Aorta, V_RA, V_RV)
    Выходы:
    - Pperi: перикардиальное давление
    - Vspt: объём межжелудочковой перегородки

    Валидация Pperi:
    - Пик должен совпадать с концом диастолы (перед P-зубцом)
    - НЕ должен быть во время систолы
    """

    def __init__(self, input_dim=5, reservoir_size=100,
                 output_dim=2, spectral_radius=0.95,
                 input_scale=0.1, sparsity=0.9):
        super().__init__()

        # Входная матрица (фиксированная)
        W_in = torch.randn(input_dim, reservoir_size) * input_scale
        self.register_buffer('W_in', W_in)

        # Матрица резервуара (фиксированная, разреженная)
        W_res = torch.randn(reservoir_size, reservoir_size)
        mask = (torch.rand(reservoir_size, reservoir_size) > sparsity).float()
        W_res = W_res * mask

        # Нормализация по спектральному радиусу
        eigenvalues = torch.linalg.eigvals(W_res).abs()
        max_eig = eigenvalues.max().item()
        if max_eig > 0:
            W_res = W_res * (spectral_radius / max_eig)
        self.register_buffer('W_res', W_res)

        # Обучаемый выходной слой
        self.readout = nn.Sequential(
            nn.Linear(reservoir_size, 32),
            nn.Tanh(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        """
        x: [batch, 5] — объёмы 5 камер сердца
        Возвращает: [batch, 2] — [Pperi, Vspt]
        """
        h = x @ self.W_in
        r = torch.tanh(h + h @ self.W_res)
        return self.readout(r)


# =============================================================
# БЛОК 5: Гибридная ODE — 5 камер сердца
# =============================================================

class HybridCardiacODE(nn.Module):
    """
    Система 5 ОДУ для объёмов камер сердца:
      dV_LA/dt  = Q_pulm_vein - Q_mitral
      dV_LV/dt  = Q_mitral    - Q_aortic
      dV_Ao/dt  = Q_aortic    - Q_systemic
      dV_RA/dt  = Q_vena_cava - Q_tricuspid
      dV_RV/dt  = Q_tricuspid - Q_pulmonary

    Давления через закон Франка-Старлинга:
      Желудочки: P = E(t) * (V - V0)
      Предсердия/аорта: P = (V - V0) / C

    Клапаны = диоды: Q > 0 только если ΔP > 0

    Reservoir Computing аппроксимирует Pperi и Vspt,
    синхронизированные с фазами ЭКГ.
    """

    def __init__(self, params, net, cardiac_activation):
        super().__init__()
        self.p = params
        self.net = net
        self.activation = cardiac_activation

    def forward(self, t, V):
        """
        V = [V_LA, V_LV, V_Aorta, V_RA, V_RV]
        Возвращает: dV/dt
        """
        V_LA, V_LV, V_Ao, V_RA, V_RV = V[0], V[1], V[2], V[3], V[4]
        p = self.p

        t_val = t.item() if isinstance(t, torch.Tensor) else t

        # ===== Активация (связь с ЭКГ) =====
        e_v = self.activation.e_ventricle(t_val)
        e_a = self.activation.e_atrium(t_val)
        e_v_t = torch.tensor(e_v, dtype=torch.float32)
        e_a_t = torch.tensor(e_a, dtype=torch.float32)

        # ===== Эластансы =====
        # Желудочки: варьирующаяся эластичность E(t)
        E_LV = p['E_lv_min'] + (p['E_lv_max'] - p['E_lv_min']) * e_v_t
        E_RV = p['E_rv_min'] + (p['E_rv_max'] - p['E_rv_min']) * e_v_t

        # Предсердия: базовая + активная компонента
        E_LA = p['E_la_min'] + (p['E_la_max'] - p['E_la_min']) * e_a_t
        E_RA = p['E_ra_min'] + (p['E_ra_max'] - p['E_ra_min']) * e_a_t

        # ===== Нейросеть: Pperi и Vspt =====
        inp = torch.stack([V_LA, V_LV, V_Ao, V_RA, V_RV]).unsqueeze(0)
        nn_out = self.net(inp).squeeze(0)
        P_peri = nn_out[0]
        V_spt = nn_out[1]

        # Коррекция объёмов через перегородку
        V_LV_free = V_LV - V_spt
        V_RV_free = V_RV + V_spt

        # ===== Давления (закон Франка-Старлинга) =====
        # Желудочки: P = E(t) * (V - V0) + Pperi
        P_LV = E_LV * (V_LV_free - p['V0_lv']) + P_peri
        P_RV = E_RV * (V_RV_free - p['V0_rv']) + P_peri

        # Предсердия: P = E(t) * (V - V0)
        P_LA = E_LA * (V_LA - p['V0_la'])
        P_RA = E_RA * (V_RA - p['V0_ra'])

        # Аорта: пассивная податливость P = V / C
        P_Ao = (V_Ao - p['V0_ao']) / p['C_ao']

        # Венозное давление (упрощённо — постоянное)
        P_venous = p['P_venous']

        # Давление в лёгочной артерии (упрощённо)
        P_pulm_artery = p['P_pulm_artery']

        # ===== Потоки через клапаны (диоды) =====

        # Митральный: LA → LV (открыт если P_LA > P_LV)
        Q_mitral = torch.clamp(
            (P_LA - P_LV) / p['R_mitral'], min=0.0)

        # Аортальный: LV → Aorta (открыт если P_LV > P_Ao)
        Q_aortic = torch.clamp(
            (P_LV - P_Ao) / p['R_aortic'], min=0.0)

        # Трикуспидальный: RA → RV (открыт если P_RA > P_RV)
        Q_tricuspid = torch.clamp(
            (P_RA - P_RV) / p['R_tricuspid'], min=0.0)

        # Пульмональный: RV → Pulm.Art. (открыт если P_RV > P_pa)
        Q_pulmonary = torch.clamp(
            (P_RV - P_pulm_artery) / p['R_pulmonary'], min=0.0)

        # Системный сток: Aorta → Veins
        Q_systemic = (P_Ao - P_venous) / p['R_systemic']

        # Возврат через вены
        Q_vena_cava = (P_venous - P_RA) / p['R_venous']

        # Лёгочный возврат в LA
        Q_pulm_vein = (P_pulm_artery - P_LA) / p['R_pulm_vein']

        # ===== Система 5 ОДУ (сохранение массы) =====
        dV = torch.zeros(5)
        dV[0] = Q_pulm_vein - Q_mitral        # dV_LA/dt
        dV[1] = Q_mitral - Q_aortic            # dV_LV/dt
        dV[2] = Q_aortic - Q_systemic          # dV_Ao/dt
        dV[3] = Q_vena_cava - Q_tricuspid      # dV_RA/dt
        dV[4] = Q_tricuspid - Q_pulmonary      # dV_RV/dt

        return dV


# =============================================================
# БЛОК 6: Верификация через ЭКГ
# =============================================================

class ECGVerifier:
    """
    Верификация модели через электромеханическое сопряжение.
    Проверяет соответствие dV/dt зубцам ЭКГ:
      P-зубец → atrial kick (малый рост dV_LV/dt)
      QRS → изоволюметрическое сокращение (dV_LV/dt ≈ 0)
      ST → выброс (dV_LV/dt < 0)
      T-зубец → изоволюметрическое расслабление (dV_LV/dt ≈ 0)
      T-P → наполнение (dV_LV/dt > 0)
    """

    def __init__(self, cardiac_activation):
        self.activation = cardiac_activation

    def extract_model_signals(self, sol_np, t_np, params):
        """
        Извлекает давления, потоки и производные из решения ODE.
        """
        V_LA = sol_np[:, 0]
        V_LV = sol_np[:, 1]
        V_Ao = sol_np[:, 2]
        V_RA = sol_np[:, 3]
        V_RV = sol_np[:, 4]

        # Эластансы
        E_LV = np.array([
            params['E_lv_min'] + (params['E_lv_max'] - params['E_lv_min'])
            * self.activation.e_ventricle(t)
            for t in t_np
        ])
        E_RV = np.array([
            params['E_rv_min'] + (params['E_rv_max'] - params['E_rv_min'])
            * self.activation.e_ventricle(t)
            for t in t_np
        ])

        # Давления
        P_LV = E_LV * (V_LV - params['V0_lv'])
        P_RV = E_RV * (V_RV - params['V0_rv'])
        P_Ao = (V_Ao - params['V0_ao']) / params['C_ao']

        # Производные объёмов
        dV_LV = np.gradient(V_LV, t_np)
        dV_RV = np.gradient(V_RV, t_np)

        # Суррогатный ЭКГ: dP_LV/dt
        dP_LV = np.gradient(P_LV, t_np)

        return {
            'V_LA': V_LA, 'V_LV': V_LV, 'V_Ao': V_Ao,
            'V_RA': V_RA, 'V_RV': V_RV,
            'P_LV': P_LV, 'P_RV': P_RV, 'P_Ao': P_Ao,
            'E_LV': E_LV, 'E_RV': E_RV,
            'dV_LV': dV_LV, 'dV_RV': dV_RV,
            'dP_LV': dP_LV,
            'ecg_surrogate': dP_LV / (np.max(np.abs(dP_LV)) + 1e-10)
        }

    def verify_phases(self, model_signals, t_np, ecg_data):
        """
        Сводная верификация: проверяет поведение dV_LV/dt
        в каждой фазе сердечного цикла по таблице.
        """
        T = self.activation.T
        T_sys = self.activation.T_sys

        results = []
        for cycle_start in np.arange(0, t_np[-1] - T, T):
            t_local = t_np - cycle_start
            mask = (t_local >= 0) & (t_local < T)
            t_c = t_local[mask]
            dV = model_signals['dV_LV'][mask]
            P = model_signals['P_LV'][mask]

            if len(t_c) < 10:
                continue

            # Фаза 1: P-зубец (предсердная систола)
            # Ожидание: малый положительный dV_LV (atrial kick)
            p_mask = (t_c > T - 0.20) & (t_c < T - 0.05)
            atrial_kick = np.mean(dV[p_mask]) if np.any(p_mask) else 0

            # Фаза 2: QRS (изоволюметрическое сокращение)
            # Ожидание: dV_LV ≈ 0, P_LV резко растёт
            qrs_mask = (t_c > 0.0) & (t_c < 0.05)
            isovol_contraction = np.mean(np.abs(dV[qrs_mask])) \
                if np.any(qrs_mask) else 0

            # Фаза 3: ST-сегмент (выброс)
            # Ожидание: dV_LV < 0 (объём падает)
            st_mask = (t_c > 0.05) & (t_c < T_sys)
            ejection = np.mean(dV[st_mask]) if np.any(st_mask) else 0

            # Фаза 4: T-зубец (изоволюметрическое расслабление)
            # Ожидание: dV_LV ≈ 0, P_LV падает
            t_wave_mask = (t_c > T_sys) & (t_c < T_sys + 0.08)
            isovol_relaxation = np.mean(np.abs(dV[t_wave_mask])) \
                if np.any(t_wave_mask) else 0

            # Фаза 5: Диастола (наполнение)
            # Ожидание: dV_LV > 0 (объём растёт)
            filling_mask = (t_c > T_sys + 0.08) & (t_c < T - 0.20)
            filling = np.mean(dV[filling_mask]) if np.any(filling_mask) else 0

            cycle_result = {
                'atrial_kick': atrial_kick,
                'atrial_kick_ok': atrial_kick > 0,
                'isovol_contraction': isovol_contraction,
                'isovol_contraction_ok': isovol_contraction < 50,
                'ejection': ejection,
                'ejection_ok': ejection < 0,
                'isovol_relaxation': isovol_relaxation,
                'isovol_relaxation_ok': isovol_relaxation < 50,
                'filling': filling,
                'filling_ok': filling > 0,
            }
            results.append(cycle_result)

        return results

    def compute_metrics(self, model_signal, real_signal):
        """Корреляция Пирсона и RMSE."""
        if len(model_signal) != len(real_signal):
            real_signal = resample(real_signal, len(model_signal))

        ms = (model_signal - np.mean(model_signal))
        rs = (real_signal - np.mean(real_signal))
        std_ms = np.std(ms)
        std_rs = np.std(rs)
        if std_ms > 1e-10:
            ms = ms / std_ms
        if std_rs > 1e-10:
            rs = rs / std_rs

        corr, p_val = pearsonr(ms, rs)
        rmse = np.sqrt(np.mean((ms - rs) ** 2))

        return {'correlation': corr, 'p_value': p_val, 'rmse': rmse}


# =============================================================
# БЛОК 7: Визуализация
# =============================================================

def plot_full_verification(ecg_data, model_signals, t_np,
                           verification_results, metrics):
    """
    8 графиков: реальное ЭКГ + модель + верификация фаз.
    """
    fig, axes = plt.subplots(4, 2, figsize=(18, 20))
    fig.suptitle(
        'Верификация гибридной модели гемодинамики через ЭКГ\n'
        'Электромеханическое сопряжение: ОДУ ↔ Reservoir Computing ↔ PhysioNet',
        fontsize=14, fontweight='bold')

    # 1. Реальное ЭКГ с R, P, T пиками
    ax = axes[0, 0]
    fs = ecg_data['fs']
    t_ecg = np.arange(len(ecg_data['normalized'])) / fs
    ax.plot(t_ecg[:int(5*fs)], ecg_data['normalized'][:int(5*fs)],
            'b-', linewidth=0.5)
    r_mask = ecg_data['r_peaks'][ecg_data['r_peaks'] < int(5*fs)]
    ax.plot(r_mask/fs, ecg_data['normalized'][r_mask],
            'rv', markersize=8, label='R-пик')
    if len(ecg_data['p_peaks']) > 0:
        p_mask = ecg_data['p_peaks'][ecg_data['p_peaks'] < int(5*fs)]
        ax.plot(p_mask/fs, ecg_data['normalized'][p_mask],
                'g^', markersize=6, label='P-зубец')
    if len(ecg_data['t_peaks']) > 0:
        t_mask = ecg_data['t_peaks'][ecg_data['t_peaks'] < int(5*fs)]
        ax.plot(t_mask/fs, ecg_data['normalized'][t_mask],
                'ms', markersize=6, label='T-зубец')
    ax.set_title(f'Реальное ЭКГ | ЧСС = {ecg_data["heart_rate"]:.0f} уд/мин')
    ax.set_xlabel('Время (с)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Средний кардиоцикл
    ax = axes[0, 1]
    mean_c = ecg_data['mean_cycle']
    t_cycle = np.linspace(0, ecg_data['rr_interval'], len(mean_c))
    for c in ecg_data['all_cycles_resampled']:
        ax.plot(t_cycle, c, 'b-', alpha=0.1, linewidth=0.5)
    ax.plot(t_cycle, mean_c, 'r-', linewidth=2.5, label='Средний цикл')
    ax.fill_between(t_cycle,
                    mean_c - ecg_data['std_cycle'],
                    mean_c + ecg_data['std_cycle'],
                    alpha=0.2, color='red', label='±1σ')
    ax.set_title(f'Кардиоциклы ({len(ecg_data["cycles"])} шт.)')
    ax.set_xlabel('Время в цикле (с)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Объёмы 5 камер
    ax = axes[1, 0]
    labels_v = ['V_LA', 'V_LV', 'V_Ao', 'V_RA', 'V_RV']
    colors_v = ['green', 'red', 'blue', 'orange', 'purple']
    for i, (lbl, clr) in enumerate(zip(labels_v, colors_v)):
        ax.plot(t_np, model_signals[lbl], color=clr, linewidth=1.5, label=lbl)
    ax.set_title('Объёмы 5 камер сердца (из ОДУ)')
    ax.set_xlabel('Время (с)')
    ax.set_ylabel('Объём (мл)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Давления
    ax = axes[1, 1]
    ax.plot(t_np, model_signals['P_LV'], 'r-', linewidth=1.5, label='P_LV')
    ax.plot(t_np, model_signals['P_RV'], 'b-', linewidth=1.5, label='P_RV')
    ax.plot(t_np, model_signals['P_Ao'], 'g-', linewidth=1.5, label='P_Ao')
    ax.set_title('Давления (закон Франка-Старлинга)')
    ax.set_xlabel('Время (с)')
    ax.set_ylabel('Давление (мм рт.ст.)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. dV_LV/dt с фазами ЭКГ
    ax = axes[2, 0]
    ax.plot(t_np, model_signals['dV_LV'], 'r-', linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    T = t_np[-1]
    ax.axvspan(0, 0.05*T, alpha=0.1, color='yellow', label='QRS (изовол.)')
    ax.axvspan(0.05*T, 0.35*T, alpha=0.1, color='red', label='Выброс')
    ax.axvspan(0.35*T, 0.43*T, alpha=0.1, color='orange', label='T (изовол.)')
    ax.axvspan(0.43*T, 0.85*T, alpha=0.1, color='green', label='Наполнение')
    ax.axvspan(0.85*T, T, alpha=0.1, color='cyan', label='P (atrial kick)')
    ax.set_title('dV_LV/dt — верификация фаз через ЭКГ')
    ax.set_xlabel('Время (с)')
    ax.set_ylabel('мл/с')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. Эластанс E(t) vs ЭКГ
    ax = axes[2, 1]
    ax.plot(t_np, model_signals['E_LV'], 'r-', linewidth=2, label='E_LV(t)')
    ax.plot(t_np, model_signals['E_RV'], 'b-', linewidth=2, label='E_RV(t)')
    ax2 = ax.twinx()
    ecg_resampled = resample(ecg_data['mean_cycle'], len(t_np))
    ax2.plot(t_np, ecg_resampled, 'g--', linewidth=1, alpha=0.7, label='ЭКГ')
    ax.set_title('Эластанс E(t) синхронизирован с ЭКГ')
    ax.set_xlabel('Время (с)')
    ax.set_ylabel('Эластанс (мм рт.ст./мл)')
    ax2.set_ylabel('ЭКГ (норм.)')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 7. Суррогатный ЭКГ vs реальный
    ax = axes[3, 0]
    surr = model_signals['ecg_surrogate']
    real_resamp = resample(ecg_data['mean_cycle'], len(surr))
    real_norm = (real_resamp - real_resamp.min()) / (real_resamp.max() - real_resamp.min() + 1e-10)
    surr_norm = (surr - surr.min()) / (surr.max() - surr.min() + 1e-10)
    ax.plot(t_np, surr_norm, 'b-', linewidth=2, label='Модель (dP_LV/dt)')
    ax.plot(t_np, real_norm, 'r--', linewidth=2, label='Реальный ЭКГ')
    ax.set_title(
        f'Сравнение | r = {metrics["correlation"]:.3f}, '
        f'RMSE = {metrics["rmse"]:.3f}')
    ax.set_xlabel('Время (с)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 8. Таблица верификации
    ax = axes[3, 1]
    ax.axis('off')
    if verification_results:
        vr = verification_results[0]
        table_data = [
            ['Фаза ЭКГ', 'Ожидание dV_LV/dt', 'Значение', 'Статус'],
            ['P-зубец', 'Рост (+)', f'{vr["atrial_kick"]:.1f}',
             '✓' if vr['atrial_kick_ok'] else '✗'],
            ['QRS', '≈ 0 (изовол.)', f'{vr["isovol_contraction"]:.1f}',
             '✓' if vr['isovol_contraction_ok'] else '✗'],
            ['ST (выброс)', 'Падение (-)', f'{vr["ejection"]:.1f}',
             '✓' if vr['ejection_ok'] else '✗'],
            ['T-зубец', '≈ 0 (изовол.)', f'{vr["isovol_relaxation"]:.1f}',
             '✓' if vr['isovol_relaxation_ok'] else '✗'],
            ['T-P (наполн.)', 'Рост (+)', f'{vr["filling"]:.1f}',
             '✓' if vr['filling_ok'] else '✗'],
        ]
        table = ax.table(cellText=table_data, loc='center',
                         cellLoc='center', colWidths=[0.25, 0.25, 0.2, 0.12])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.8)
        # Раскраска заголовка
        for j in range(4):
            table[0, j].set_facecolor('#4472C4')
            table[0, j].set_text_props(color='white', fontweight='bold')
        # Раскраска статуса
        for i in range(1, 6):
            color = '#C6EFCE' if table_data[i][3] == '✓' else '#FFC7CE'
            table[i, 3].set_facecolor(color)
    ax.set_title('Сводная таблица верификации', fontsize=12, pad=20)

    plt.tight_layout()
    plt.savefig('cardiac_verification_result.png', dpi=150,
                bbox_inches='tight')
    plt.show()


# =============================================================
# БЛОК 8: Главная программа
# =============================================================

def main():
    print("=" * 65)
    print("  ГИБРИДНАЯ МОДЕЛЬ ГЕМОДИНАМИКИ + RESERVOIR COMPUTING")
    print("  Верификация через реальное ЭКГ (PhysioNet)")
    print("  5 ОДУ ↔ Электромеханическое сопряжение ↔ ЭКГ")
    print("=" * 65)

    # ===== 1. Загрузка ЭКГ =====
    print("\n[1/6] Загрузка ЭКГ с PhysioNet...")
    loader = PhysioNetECGLoader()
    try:
        person_id = int(input("Номер пациента (1-90, Enter=1): ") or "1")
        record_id = int(input("Номер записи (1-20, Enter=1): ") or "1")
    except ValueError:
        person_id, record_id = 1, 1

    edf_path = loader.download_record(person_id, record_id)
    if edf_path and os.path.exists(edf_path):
        ecg_raw, fs = loader.read_edf(edf_path)
    else:
        print("Файл не найден. Генерирую синтетическое ЭКГ...")
        fs = 500
        ecg_raw = _generate_synthetic_ecg(fs, duration=10)

    # ===== 2. Предобработка =====
    print("\n[2/6] Предобработка ЭКГ...")
    preprocessor = ECGPreprocessor()
    ecg_data = preprocessor.full_pipeline(ecg_raw, fs, target_length=500)
    if ecg_data is None:
        print("Ошибка обработки. Выход.")
        return

    # ===== 3. Параметры модели из ЭКГ =====
    print("\n[3/6] Настройка модели по данным ЭКГ...")
    rr = ecg_data['rr_interval']
    T_sys = 0.3 * rr / 0.85  # масштабирование систолы

    params = {
        # Эластансы желудочков
        'E_lv_min': torch.tensor(0.08),
        'E_lv_max': torch.tensor(2.5),
        'E_rv_min': torch.tensor(0.05),
        'E_rv_max': torch.tensor(1.15),
        # Эластансы предсердий
        'E_la_min': torch.tensor(0.15),
        'E_la_max': torch.tensor(0.25),
        'E_ra_min': torch.tensor(0.10),
        'E_ra_max': torch.tensor(0.15),
        # Податливость аорты
        'C_ao': torch.tensor(1.5),
        # Ненагруженные объёмы
        'V0_lv': torch.tensor(5.0),
        'V0_rv': torch.tensor(10.0),
        'V0_la': torch.tensor(4.0),
        'V0_ra': torch.tensor(4.0),
        'V0_ao': torch.tensor(250.0),
        # Сопротивления клапанов
        'R_mitral': torch.tensor(0.01),
        'R_aortic': torch.tensor(0.01),
        'R_tricuspid': torch.tensor(0.01),
        'R_pulmonary': torch.tensor(0.01),
        # Сосудистые сопротивления
        'R_systemic': torch.tensor(1.0),
        'R_venous': torch.tensor(0.05),
        'R_pulm_vein': torch.tensor(0.08),
        # Постоянные давления
        'P_venous': torch.tensor(5.0),
        'P_pulm_artery': torch.tensor(15.0),
    }

    cardiac_act = CardiacActivation(
        T=rr,
        T_sys=T_sys,
        T_atrial=0.1,
        atrial_delay=0.12
    )

    print(f"  Период RR: {rr*1000:.0f} мс")
    print(f"  Систола: {T_sys*1000:.0f} мс")
    print(f"  ЧСС: {ecg_data['heart_rate']:.0f} уд/мин")

    # ===== 4. Решение ОДУ =====
    print(f"\n[4/6] Решение системы 5 ОДУ...")

    net = ReservoirNet(input_dim=5, reservoir_size=100, output_dim=2)
    ode_func = HybridCardiacODE(params, net, cardiac_act)

    # Начальные условия: [V_LA, V_LV, V_Ao, V_RA, V_RV]
    V0 = torch.tensor([30.0, 120.0, 280.0, 25.0, 110.0])
    t_span = torch.linspace(0.0, rr, 500, dtype=torch.float32)

    with torch.no_grad():
        sol = odeint(ode_func, V0, t_span, method='bosh3')

    sol_np = sol.detach().cpu().numpy()
    t_np = t_span.numpy()
    print("  Решение получено!")

    # ===== 5. Верификация =====
    print("\n[5/6] Верификация через ЭКГ...")

    params_np = {k: v.item() for k, v in params.items()}
    verifier = ECGVerifier(cardiac_act)
    model_signals = verifier.extract_model_signals(sol_np, t_np, params_np)
    verification = verifier.verify_phases(model_signals, t_np, ecg_data)
    metrics = verifier.compute_metrics(
        model_signals['ecg_surrogate'], ecg_data['mean_cycle'])

    # ===== 6. Визуализация =====
    print("\n[6/6] Визуализация...")
    plot_full_verification(ecg_data, model_signals, t_np,
                           verification, metrics)

    # Итоговый отчёт
    print("\n" + "=" * 65)
    print("  РЕЗУЛЬТАТЫ ВЕРИФИКАЦИИ")
    print("=" * 65)
    print(f"  Пациент: Person_{person_id:02d}, запись rec_{record_id}")
    print(f"  ЧСС: {ecg_data['heart_rate']:.1f} уд/мин")
    print(f"  Корреляция модель/ЭКГ: {metrics['correlation']:.4f}")
    print(f"  RMSE модель/ЭКГ: {metrics['rmse']:.4f}")

    if verification:
        vr = verification[0]
        print(f"\n  Фазовая верификация:")
        phases = [
            ('P-зубец (atrial kick)', vr['atrial_kick_ok']),
            ('QRS (изоволюметр. сокращ.)', vr['isovol_contraction_ok']),
            ('ST (выброс)', vr['ejection_ok']),
            ('T (изоволюметр. расслабл.)', vr['isovol_relaxation_ok']),
            ('T-P (наполнение)', vr['filling_ok']),
        ]
        for name, ok in phases:
            status = '✓' if ok else '✗'
            print(f"    {status} {name}")

    print(f"\n  График: cardiac_verification_result.png")
    print("=" * 65)


def _generate_synthetic_ecg(fs=500, duration=10):
    """Синтетическое ЭКГ если PhysioNet недоступен."""
    t = np.arange(0, duration, 1.0 / fs)
    period = 60.0 / 72
    ecg = np.zeros_like(t)
    for i, ti in enumerate(t):
        phase = (ti % period) / period
        ecg[i] += 0.15 * np.exp(-((phase - 0.10)**2) / 0.001)  # P
        ecg[i] -= 0.10 * np.exp(-((phase - 0.22)**2) / 0.0002)  # Q
        ecg[i] += 1.00 * np.exp(-((phase - 0.25)**2) / 0.0003)  # R
        ecg[i] -= 0.20 * np.exp(-((phase - 0.28)**2) / 0.0002)  # S
        ecg[i] += 0.30 * np.exp(-((phase - 0.45)**2) / 0.003)   # T
    ecg += np.random.normal(0, 0.02, len(ecg))
    return ecg


if __name__ == '__main__':
    main()
