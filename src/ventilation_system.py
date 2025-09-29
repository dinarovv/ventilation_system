import numpy as np
import matplotlib.pyplot as plt

from tsukamoto_model import trapmf


class VentilationSystem:
    
    def __init__(self, temp_range=(0, 101, 1), hum_range=(0, 101, 1), fan_range=(0, 101, 1)):
        self.temp_range = temp_range
        self.hum_range = hum_range
        self.fan_range = fan_range

        self.x1 = temp_range[0]
        self.x2 = temp_range[1] - 1
        self.res = 0

        self.temperature = np.linspace(temp_range[0], temp_range[1], 1000)
        self.humidity = np.linspace(hum_range[0], hum_range[1], 1000)
        self.fan_speed = np.linspace(fan_range[0], fan_range[1], 1000)

        self.temp_mfs = {}
        self.hum_mfs = {}
        self.fan_mfs = {}
        self.rules = []

        self.set_default_mfs()
        self.set_rules()

    def set_trapmf(self, universe, params):
        return lambda x: trapmf(np.array(x), *params)

    def set_default_mfs(self):
        tr = self.temp_range
        temp_min, temp_max = tr[0], tr[1]
        temp_span = temp_max - temp_min
        

        self.temp_mfs = {
            'very_low': self.set_trapmf(self.temperature, [
                temp_min - temp_max ** 4 if temp_max > 10 else temp_min - (abs(temp_max) + 10) ** 4,
                temp_min,
                temp_min + 0.2 * temp_span,
                temp_min + 0.3 * temp_span
            ]),
            'low': self.set_trapmf(self.temperature, [
                temp_min + 0.2 * temp_span,
                temp_min + 0.3 * temp_span,
                temp_min + 0.4 * temp_span,
                temp_min + 0.5 * temp_span
            ]),
            'medium': self.set_trapmf(self.temperature, [
                temp_min + 0.4 * temp_span,
                temp_min + 0.5 * temp_span,
                temp_min + 0.6 * temp_span,
                temp_min + 0.7 * temp_span
            ]),
            'high': self.set_trapmf(self.temperature, [
                temp_min + 0.6 * temp_span,
                temp_min + 0.7 * temp_span,
                temp_min + 0.8 * temp_span,
                temp_min + 0.9 * temp_span
            ]),
            'very_high': self.set_trapmf(self.temperature, [
                temp_min + 0.8 * temp_span,
                temp_min + 0.9 * temp_span,
                temp_max,
                temp_max * 10
            ]),
        }

        self.hum_mfs = {
            'very_low': self.set_trapmf(self.humidity, [-100, 0, 20, 30]),
            'low': self.set_trapmf(self.humidity, [20, 30, 40, 50]),
            'medium': self.set_trapmf(self.humidity, [40, 50, 60, 70]),
            'high': self.set_trapmf(self.humidity, [60, 70, 80, 90]),
            'very_high': self.set_trapmf(self.humidity, [80, 90, 100, 1000]),
        }

        self.fan_mfs = {
            'very_low': lambda mu: self.defuzz_single(mu, [-100, 0, 20, 30], ascending=True),
            'low': lambda mu: self.defuzz_single(mu, [20, 30, 40, 50], ascending=True),
            'medium': lambda mu: self.defuzz_single(mu, [40, 50, 60, 70], ascending=True),
            'high': lambda mu: self.defuzz_single(mu, [60, 70, 80, 90], ascending=True),
            'very_high': lambda mu: self.defuzz_single(mu, [80, 90, 100, 1000], ascending=True),
        }

    def defuzz_single(self, mu_target, params, ascending=True):
        z_values = self.fan_speed
        mf_values = trapmf(z_values, *params)

        if ascending:
            idx = np.where(mf_values >= mu_target - 1e-3)[0]
            if len(idx) == 0:
                return np.mean(z_values)
            return z_values[idx[0]]
        else:
            idx = np.where(mf_values >= mu_target - 1e-3)[0]
            if len(idx) == 0:
                return np.mean(z_values)
            return z_values[idx[-1]]

    def set_rules(self):
        self.rules = [
            # Very Low Temp: prioritize humidity control
            ('very_low', 'very_low', 'very_low'),
            ('very_low', 'low', 'very_low'),
            ('very_low', 'medium', 'low'),
            ('very_low', 'high', 'high'),       # Усилено для борьбы с влажностью
            ('very_low', 'very_high', 'high'),  # Усилено для борьбы с влажностью
            
            # Low Temp: still cautious but more responsive to humidity
            ('low', 'very_low', 'very_low'),
            ('low', 'low', 'low'),
            ('low', 'medium', 'low'),
            ('low', 'high', 'medium'),
            ('low', 'very_high', 'high'),       # Усилено
            
            # Medium Temp: balance
            ('medium', 'very_low', 'low'),
            ('medium', 'low', 'low'),
            ('medium', 'medium', 'medium'),
            ('medium', 'high', 'high'),
            ('medium', 'very_high', 'high'),
            
            # High Temp: prioritize cooling
            ('high', 'very_low', 'high'),       # Усилено из-за температуры
            ('high', 'low', 'high'),
            ('high', 'medium', 'high'),
            ('high', 'high', 'very_high'),
            ('high', 'very_high', 'very_high'),
            
            # Very High Temp: MAX COOLING
            ('very_high', 'very_low', 'very_high'),  # Критично при любой влажности
            ('very_high', 'low', 'very_high'),
            ('very_high', 'medium', 'very_high'),
            ('very_high', 'high', 'very_high'),
            ('very_high', 'very_high', 'very_high'),
        ]

    def evaluate(self, temp_val, hum_val):
        numerator = 0
        denominator = 0

        for temp_term, hum_term, fan_term in self.rules:
            mu_temp = float(self.temp_mfs[temp_term](np.array([temp_val])))
            mu_hum = float(self.hum_mfs[hum_term](np.array([hum_val])))
            alpha = min(mu_temp, mu_hum)

            z = self.fan_mfs[fan_term](alpha)
            numerator += alpha * z
            denominator += alpha

        return numerator / denominator if denominator != 0 else 0

    def visualize(self, temp_val, hum_val, fan_val):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

        for name, mf in self.temp_mfs.items():
            y = mf(self.temperature)
            ax1.plot(self.temperature, y, label=name, linewidth=2)
            ax1.scatter(temp_val, mf(np.array([temp_val])), color='black')
        ax1.set_title('Температура')
        ax1.set_ylim(0, 1.05)
        ax1.legend(loc="upper right")
        ax1.grid()

        for name, mf in self.hum_mfs.items():
            y = mf(self.humidity)
            ax2.plot(self.humidity, y, label=name, linewidth=2)
            ax2.scatter(hum_val, mf(np.array([hum_val])), color='black')
        ax2.set_title('Влажность')
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc="upper right")
        ax2.grid()

        for name in self.fan_mfs:
            y = trapmf(self.fan_speed, *self.get_fan_params(name))
            ax3.plot(self.fan_speed, y, label=name, linewidth=2)
            
        ax3.axvline(fan_val, color='red', linestyle='--')
        ax3.scatter(fan_val, 1, color='red', zorder=5)
        ax3.annotate(f'Рекомендуемая: {fan_val:.1f}%', (fan_val, 1), textcoords="offset points", xytext=(10, 10), ha='left', color='red', fontsize=10, fontweight='bold')
        ax3.set_title('Скорость вентиляции')
        ax3.set_ylim(0, 1.05)
        ax3.legend(loc="upper right")
        ax3.grid()

        plt.tight_layout()
        plt.show()

    def get_fan_params(self, term):
        params_map = {
            'very_low': [-100, 0, 20, 30],
            'low': [20, 30, 40, 50],
            'medium': [40, 50, 60, 70],
            'high': [60, 70, 80, 90],
            'very_high': [80, 90, 100, 1000],
        }
        return params_map[term]

    def hello(self):
        print("=== Система нечеткого управления вентиляцией ===")
        print("=== Модель Цукамото с трапециевидными функциями ===\n")

    def input_temp_range(self):
        while True:
            try:
                self.x1, self.x2 = map(int, input('Укажите диапазон температур через пробел (пример: -30 30): ').split())
                self.temp_range = (self.x1, self.x2 + 1, 1)
                self.temperature = np.linspace(self.temp_range[0], self.temp_range[1], 1000)
                self.set_default_mfs()
                return
            except:
                print('Неверный ввод! Попробуйте еще раз..')

    def input_temp(self):
        temp = int(input(f'Введите значение температуры [{self.x1};{self.x2}]: '))
        if not (self.x1 <= temp <= self.x2):
            raise ValueError
        return temp
    
    def input_hum(self):
        hum = int(input('Введите значение влажности [0;100]: '))
        if not (0 <= hum <= 100):
            raise ValueError
        return hum
        
    def input_values(self):
        while True:
            try:
                temp = self.input_temp()
                break
            except Exception:
                print('Неверный ввод! Попробуйте еще раз..')

        while True:
            try:
                hum = self.input_hum()
                break
            except Exception:
                print('Неверный ввод! Попробуйте еще раз..')

        return temp, hum

    def show_res(self):
        print(f'\nРекомендуемая мощность вентилирования: {self.res:.2f}%\n')

    def is_very_high(self, temp):
        if  temp >= int(self.temp_range[0] + 0.9 * (self.temp_range[1] - self.temp_range[0])):
            self.res = 100
        return

    def run(self):
        self.hello()
        self.input_temp_range()
        temp, hum = self.input_values()
        self.res = self.evaluate(temp, hum)
        self.is_very_high(temp)
        self.show_res()
        self.visualize(temp, hum, self.res)