import pickle
from datetime import datetime
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from pprint import pprint
from sklearn.metrics import r2_score, mean_squared_error

from main import create_all_features, abc_transform, append_to_csv

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")


def add_abc(abc_params_dict, data_wo_abc):
    """
    Применяет ABC преобразования к данным
    abc_params_dict: словарь вида {'channel_A': value, 'channel_B': value, 'channel_C': value}
    data_wo_abc: DataFrame с исходными данными
    """
    channels = set()
    for key in abc_params_dict.keys():
        if key.endswith(('_A', '_B', '_C')):
            channel = key.rsplit('_', 1)[0]  # Убираем _A, _B, _C
            channels.add(channel)

    applied_params = {}

    for channel in channels:
        a_key = f'{channel}_A'
        b_key = f'{channel}_B'
        c_key = f'{channel}_C'

        if all(key in abc_params_dict for key in [a_key, b_key, c_key]):
            A = abc_params_dict[a_key]
            B = abc_params_dict[b_key]
            C = abc_params_dict[c_key]

            print(f'{channel}_abc', 'qq')

            # Ищем исходную колонку (убираем возможные суффиксы)
            original_feature = channel.lower()

            if original_feature in data_wo_abc.columns:
                # Применяем ABC преобразование
                original_values = data_wo_abc[original_feature].values
                abc_transformed = abc_transform(original_values, A, B, C)
                feature_name = f'{original_feature}_abc'
                data_wo_abc[feature_name] = abc_transformed
                applied_params[a_key] = A
                applied_params[b_key] = B
                applied_params[c_key] = C

            else:
                print(f"Исходная фича не найдена: {original_feature}")

    return applied_params, data_wo_abc


# Использование:
# abc_params_dict = {'federal_tv_A': 0.8, 'federal_tv_B': 7.5, 'federal_tv_C': 2.2, ...}
# applied_params, data_with_abc = add_abc(abc_params_dict, data_wo_abc)


def normalize_feature_name(feature_name):
    """
    Нормализует название признака для поиска соответствий
    """
    # Убираем пробелы, запятые, точки и приводим к нижнему регистру
    normalized = (feature_name.lower()
                  .replace(' ', '_')
                  .replace(',', '')
                  .replace('.', '')
                  .replace('(', '')
                  .replace(')', ''))
    return normalized


class MMMVisualizer:
    def __init__(self, coef, features, abc_params):
        """
        results_df: DataFrame с колонками features, p-values, coef, A, B, C
        """
        self.coef = np.array(coef)
        features = list(map(lambda x: x.lower(), features))
        self.features = features
        self.abc_params = abc_params

    def plot_saturation_curves(self, save_path='saturation_curves.png'):
        """
        Строит кривые насыщения для каждого медиа-канала
        model_abc_params: словарь с ABC параметрами вида {'channel_A': value, 'channel_B': value, 'channel_C': value}
        """
        model_abc_params = self.abc_params
        # Извлекаем уникальные каналы из ключей
        channels = set()
        for key in model_abc_params.keys():
            if key.endswith(('_A', '_B', '_C')):
                channel = key.rsplit('_', 1)[0]  # Убираем _A, _B, _C
                channels.add(channel)

        media_channels = []
        for channel in channels:
            # Проверяем что есть все три параметра
            a_key = f'{channel}_A'
            b_key = f'{channel}_B'
            c_key = f'{channel}_C'

            if all(key in model_abc_params for key in [a_key, b_key, c_key]):
                media_channels.append({
                    'name': channel,
                    'A': model_abc_params[a_key],
                    'B': model_abc_params[b_key],
                    'C': model_abc_params[c_key]
                })

        n_channels = len(media_channels)
        if n_channels == 0:
            print("Нет медиа-каналов для визуализации")
            return

        cols = min(3, n_channels)
        rows = (n_channels + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

        if n_channels == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if cols == 1 else list(axes)
        else:
            axes = axes.flatten()

        fig.suptitle('Кривые насыщения медиа-каналов', fontsize=16, fontweight='bold')

        for idx, channel_data in enumerate(media_channels):
            ax = axes[idx]

            A = channel_data['A']
            B = channel_data['B']
            C = channel_data['C']
            channel_name = channel_data['name'].replace('_', ' ').title()

            max_spend = 1000
            spend_range = np.linspace(0, max_spend, 200)

            def abc_curve(spend, a, b, c):
                scaled = b * spend
                saturated = (c * scaled) / (1 + c * scaled)
                return saturated

            response = abc_curve(spend_range, A, B, C)

            ax.plot(spend_range, response, linewidth=3, color='#2E86AB', label='Кривая отклика')
            ax.fill_between(spend_range, 0, response, alpha=0.3, color='#2E86AB')

            current_spend = 500
            current_response = abc_curve(current_spend, A, B, C)
            ax.scatter([current_spend], [current_response], color='red', s=100,
                       zorder=5, label='Текущий уровень')

            ax.set_title(f'{channel_name}\nA={A:.3f}, B={B:.3f}, C={C:.3f}',
                         fontweight='bold', pad=20)
            ax.set_xlabel('Медиа-инвестиции', fontweight='bold')
            ax.set_ylabel('Прирост продаж', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()

            if C < 1:
                saturation_text = "Быстрое насыщение"
            elif C < 2:
                saturation_text = "Умеренное насыщение"
            else:
                saturation_text = "Медленное насыщение"

            ax.text(0.05, 0.95, saturation_text, transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    verticalalignment='top', fontsize=9)

        # Убираем лишние подграфики
        for idx in range(n_channels, len(axes)):
            axes[idx].remove()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    # Использование:
    # plot_saturation_curves(model_abc_params)

    def create_model_from_coefficients(self, train_data, forecast_data, abc_params, target_col='sales',
                                       intercept=None, save_path=None):
        """
        Создает модель на основе готовых коэффициентов и ABC параметров

        Args:
            original_data: исходные данные (DataFrame)
            coefficients_df: DataFrame с колонками features, p-values, coef, A, B, C
            target_col: название целевой переменной
            intercept: свободный член (если None, рассчитается автоматически)
            save_path: путь для сохранения модели (опционально)

        Returns:
            dict: model_data с моделью, преобразованными данными и метаданными
        """
        features = self.features
        print(train_data.columns)
        print(features)
        missing_features = [f for f in features if f not in train_data.columns]

        if missing_features:
            print(f"Отсутствующие признаки: {missing_features}")
            return None

        clean_data = train_data[features + [target_col]].dropna()
        X = clean_data[features].values
        y = clean_data[target_col].values
        coefficients = self.coef

        print(f"Подготовлено данных: {len(clean_data)} наблюдений, {len(features)} признаков")

        model = LinearRegression()
        model.coef_ = coefficients
        if intercept is None:
            predicted_without_intercept = X @ coefficients
            intercept = np.mean(y) - np.mean(predicted_without_intercept)

        model.intercept_ = intercept
        model.n_features_in_ = len(features)
        model.feature_names_in_ = np.array(features)

        y_pred = model.predict(X)
        r2_score_ = r2_score(y, y_pred)

        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)

        print(f"Метрики на тренировочных данных:")
        print(f"   R²: {r2_score_:.4f}")
        print(f"   RMSE: {rmse:.2f}")

        params = abc_params.copy()
        used_media = [f for f in features if '_abc' in f]
        used_price = [f for f in features if any(p in f.lower() for p in ['price', 'premium', 'discount'])]
        used_seasonal = [f for f in features if
                         any(s in f.lower() for s in ['spring', 'summer', 'autumn', 'winter', 'holiday', 'trend'])]
        used_other = [f for f in features if f not in used_media + used_price + used_seasonal]
        params.update({
            'used_media': used_media,
            'used_price': used_price,
            'used_seasonal': used_seasonal,
            'used_other': used_other,
            'intercept': intercept
        })

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata = {
            'timestamp': timestamp,
            'creation_method': 'from_coefficients',
            'n_features': len(features),
            'target_col': target_col,
            'train_data_shape': clean_data.shape,
            'r2_score': r2_score_,
            'rmse': rmse,
            'intercept': intercept,
            'feature_groups': {
                'media': used_media,
                'price': used_price,
                'seasonal': used_seasonal,
                'other': used_other
            },
            'abc_transformations': len([f for f in features if '_abc' in f])
        }

        model_data = {
            'model': model,
            'features': features,
            'params': params,
            'transformed_data': clean_data,
            'coefficients_df': self.coef,
            'metadata': metadata,
            'abc_params': abc_params
        }
        result_predict = self.predict_with_model(model_data, forecast_data)
        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Модель сохранена: {save_path}")
        return model_data, result_predict

    def plot_real_contribution_over_time(self, df_data, model, save_path='real_contribution.png'):
        """
        Реальная декомпозиция продаж по времени
        df_data: DataFrame с временными данными и фичами
        model: обученная модель
        features: список фич, которые использует модель
        """
        features = self.features

        # Проверяем что все фичи есть в данных
        missing_features = [f for f in features if f.replace('_abc', '') not in df_data.columns]
        if missing_features:
            print(f"Отсутствующие фичи: {missing_features}")
            return

        # Подготавливаем данные
        X = df_data[features].values
        coefficients = model.coef_
        intercept = model.intercept_

        # Рассчитываем вклад каждой фичи для каждого периода
        contributions = {}

        # Константа (базовый уровень)
        contributions['Константа'] = [intercept] * len(df_data)

        # Вклад каждой фичи = значение_фичи * коэффициент
        for i, feature in enumerate(features):
            feature_contribution = df_data[feature].values * coefficients[i]
            clean_name = feature.replace('_abc', '').replace('_', ' ').title()
            contributions[clean_name] = feature_contribution

        # Временная ось
        if 'Week' in df_data.columns:
            x = df_data['Week']
            x_label = 'Недели'
        else:
            x = range(len(df_data))
            x_label = 'Период'

        # Подготавливаем данные для stackplot
        labels = list(contributions.keys())
        y_data = list(contributions.values())

        # Цвета
        mycolors = ['lightblue', 'tab:green', 'tab:red', 'tab:orange', 'tab:brown',
                    'tab:pink', 'tab:olive', 'tab:cyan', 'tab:purple', 'tab:gray',
                    'darkblue', 'darkgreen', 'darkred', 'darkorange']

        # Обрезаем под количество фичей
        colors = mycolors[:len(labels)]

        # Создаем график
        fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=80)

        # Строим stackplot
        ax.stackplot(x, *y_data, labels=labels, colors=colors, alpha=0.8)

        # Добавляем линию фактических продаж для сравнения
        if 'Sales' in df_data.columns:
            actual_sales = df_data['Sales'].values
            predicted_sales = np.sum(y_data, axis=0)

            ax.plot(x, actual_sales, 'k-', linewidth=3, label='Фактические продажи', alpha=0.9)
            ax.plot(x, predicted_sales, 'r--', linewidth=2, label='Модельный прогноз', alpha=0.9)

        # Оформление в стиле примера
        ax.set_title('Декомпозиция продаж по факторам во времени\n(Реальные данные)',
                     fontsize=18, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel('Продажи', fontsize=14)

        # Легенда
        ax.legend(fontsize=10, ncol=3, loc='upper left')

        # Форматирование осей
        if len(x) > 20:
            step = len(x) // 10
            plt.xticks(x[::step], fontsize=10, rotation=45)
        else:
            plt.xticks(x, fontsize=10, rotation=45)

        # Форматирование оси Y
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:,.0f}'))

        # Стиль как в примере - облегчаем границы
        plt.gca().spines["top"].set_alpha(0)
        plt.gca().spines["bottom"].set_alpha(.3)
        plt.gca().spines["right"].set_alpha(0)
        plt.gca().spines["left"].set_alpha(.3)

        # Сетка
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        # Выводим статистику вкладов
        print(f"\n СРЕДНИЕ ВКЛАДЫ ПО ФАКТОРАМ:")

        total_avg_contribution = {}
        for name, values in contributions.items():
            avg_contribution = np.mean(values)
            total_avg_contribution[name] = avg_contribution
            print(f"{name:<25}: {avg_contribution:>10,.0f}")

        total = sum(total_avg_contribution.values())
        print(f"{'ИТОГО':<25}: {total:>10,.0f}")

        if 'Sales' in df_data.columns:
            actual_avg = df_data['Sales'].mean()
            print(f"{'Факт (среднее)':<25}: {actual_avg:>10,.0f}")
            print(f"{'Отклонение':<25}: {total - actual_avg:>10,.0f}")

    def plot_feature_elasticity(self, model, transformed_data, save_path='feature_elasticity.png'):
        """
        График эластичности факторов (% изменение Y при изменении X на 1%)
        """
        features = self.features
        coefficients = model.coef_

        elasticities = []

        target_col = self.target_col if hasattr(self, 'target_col') else 'sales'
        mean_y = np.mean(transformed_data[target_col])

        for i, feature in enumerate(features):
            if feature in transformed_data.columns:
                coef = coefficients[i]
                mean_x = np.mean(transformed_data[feature])

                # Эластичность = (dY/dX) * (X/Y) = coef * (mean_X / mean_Y)
                if mean_y != 0 and mean_x != 0:
                    elasticity = coef * (mean_x / mean_y)

                    clean_name = feature.replace('_abc', '').replace('_', ' ').title()
                    elasticities.append((clean_name, elasticity, coef))

        # Сортируем по абсолютной эластичности
        elasticities.sort(key=lambda x: abs(x[1]), reverse=True)

        names = [x[0] for x in elasticities]
        elast_values = [x[1] for x in elasticities]
        colors = ['tab:green' if v > 0 else 'tab:red' for v in elast_values]

        fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=80)

        bars = ax.barh(names, [abs(v) for v in elast_values], color=colors, alpha=0.8)

        for i, (bar, val) in enumerate(zip(bars, elast_values)):
            width = bar.get_width()
            ax.text(width + max([abs(v) for v in elast_values]) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:+.3f}', ha='left', va='center', fontweight='bold')

        ax.set_title('Эластичность факторов\n(% изменение продаж при изменении фактора на 1%)',
                     fontsize=16, fontweight='bold')
        ax.set_xlabel('|Эластичность|')

        ax.spines["top"].set_alpha(0)
        ax.spines["bottom"].set_alpha(.3)
        ax.spines["right"].set_alpha(0)
        ax.spines["left"].set_alpha(.3)
        ax.grid(True, alpha=0.3, axis='x')

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='tab:green', label='Положительная эластичность'),
            Patch(facecolor='tab:red', label='Отрицательная эластичность')
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return elasticities

    def predict_with_model(self, model_data, new_data, target_col='sales'):
        """
        Делает прогноз с созданной моделью

        Args:
            model_data: результат create_model_from_coefficients
            new_data: новые данные для прогноза (DataFrame)
            target_col: название целевой переменной

        Returns:
            dict: прогнозы и метрики качества
        """

        print(" Прогнозирование с моделью...")

        # Извлекаем компоненты модели
        model = model_data['model']
        features = model_data['features']
        abc_params = model_data['abc_params']

        # Подготавливаем новые данные
        forecast_data = new_data.copy()

        # Проверяем наличие всех необходимых признаков
        missing_features = [f for f in features if f not in forecast_data.columns]

        if missing_features:

            # Пытаемся продолжить с доступными признаками
            available_features = [f for f in features if f in forecast_data.columns]
            if len(available_features) < len(features) * 0.5:  # Меньше 50% признаков
                return None

            print(f"️ Продолжаем с {len(available_features)}/{len(features)} признаками")
            # Нужно пересоздать модель с доступными признаками
            features = available_features

            # Фильтруем коэффициенты
            available_indices = [i for i, feature in enumerate(self.features) if feature in available_features]

            # Фильтруем коэффициенты
            filtered_coefficients = [self.coef[i] for i in available_indices]

            # Создаем новую модель
            model = LinearRegression()
            model.coef_ = filtered_coefficients
            model.intercept_ = model_data['model'].intercept_  # Используем тот же intercept
            model.n_features_in_ = len(features)
            model.feature_names_in_ = np.array(features)

        # Заменяем NaN на 0 (как в обучении)
        numeric_columns = forecast_data.select_dtypes(include=[np.number]).columns
        forecast_data[numeric_columns] = forecast_data[numeric_columns].fillna(0)
        forecast_data[numeric_columns] = forecast_data[numeric_columns].replace([np.inf, -np.inf], 0)

        # Извлекаем признаки
        try:
            X_forecast = forecast_data[features].values
        except KeyError as e:
            return None

        # Делаем прогноз
        print(f" Выполняем прогноз для {len(X_forecast)} наблюдений...")

        try:
            predictions = model.predict(X_forecast)
        except Exception as e:
            return None

        # Результат
        result = {
            'predictions': predictions,
            'forecast_data': forecast_data,
            'used_features': features,
            'missing_features': missing_features,
            'n_predictions': len(predictions),
            'prediction_stats': {
                'mean': np.mean(predictions),
                'std': np.std(predictions),
                'min': np.min(predictions),
                'max': np.max(predictions)
            }
        }

        # Если есть фактические значения - считаем метрики
        if target_col in forecast_data.columns:
            actual = forecast_data[target_col].values

            # Убираем NaN из сравнения
            valid_mask = ~(np.isnan(actual) | np.isnan(predictions))
            if np.sum(valid_mask) > 0:
                actual_clean = actual[valid_mask]
                pred_clean = predictions[valid_mask]
                mse = mean_squared_error(actual_clean, pred_clean)
                r2 = r2_score(actual_clean, pred_clean)
                print('Метрики на тестовых данных:')
                print(actual_clean, pred_clean)
                print('rmse', np.sqrt(mse))
                print('r2', r2)
        # print(forecast_data, predictions)
        # self.save_forecast_results(result)
        return result


model_coef = [532.7297892684238, 120.03454187651681, -15.770825397486512, 825.4161500758993]

model_features = ['federal_tv_abc', 'Реклама_в_прессе_руб_abc', 'trend', 'is_holiday_season']

model_abc_params = {'federal_tv_A': 0.8008045671400532,
                    'federal_tv_B': 7.543478384254816,
                    'federal_tv_C': 2.2158660963719794,

                    'Реклама_в_прессе_руб_A': 0.4808128913025257,
                    'Реклама_в_прессе_руб_B': 9.738981706691094,
                    'Реклама_в_прессе_руб_C': 4.8761879232236405
                    }

# params = pd.read_excel('model_params.xlsx')

data = pd.read_csv('data.csv', sep=';', encoding='utf-8')
data['Week'] = pd.to_datetime(data['Week'], format='%d.%m.%Y')

data = create_all_features(data.copy())
normalized_columns = list(map(lambda x: normalize_feature_name(x), data.columns))
data.columns = normalized_columns

train_end = pd.to_datetime('2012-06-30')
forecast_end = pd.to_datetime('2012-12-30')

dict_params, data = add_abc(model_abc_params, data)

data.to_csv('data_abc.csv', index=False)

train_data = data[data['week'] <= train_end].copy()
forecast_data = data[(data['week'] > train_end) & (data['week'] <= forecast_end)].copy()

visualizer = MMMVisualizer(model_coef, model_features, model_abc_params)

model, result_predict = visualizer.create_model_from_coefficients(train_data, forecast_data, dict_params)

# Строим все графики

visualizer.plot_saturation_curves()
visualizer.plot_real_contribution_over_time(train_data, model['model'])
visualizer.plot_feature_elasticity(model['model'], train_data)

# print(model)
