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
                print(feature_name, 'feature_name')
                data_wo_abc[feature_name] = abc_transformed
                applied_params[a_key] = A
                applied_params[b_key] = B
                applied_params[c_key] = C

            else:
                print(f"Исходная фича не найдена: {original_feature}")

    return applied_params, data_wo_abc


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
    def __init__(self, coef, features, abc_params, train_data):
        """
        results_df: DataFrame с колонками features, p-values, coef, A, B, C
        """
        self.train_data = train_data
        self.coef = np.array(coef)
        features = list(map(lambda x: x.lower(), features))
        self.features = features
        self.abc_params = abc_params

    def plot_saturation_curves(self, save_path='saturation_curves.png'):
        """
        Строит кривые насыщения с оптимизированными диапазонами
        """
        model_abc_params = self.abc_params

        # Получаем реальные данные
        real_media_data = self.get_real_media_levels()

        # Извлекаем уникальные каналы из ключей
        channels = set()
        for key in model_abc_params.keys():
            if key.endswith(('_A', '_B', '_C')):
                channel = key.rsplit('_', 1)[0]
                channels.add(channel)

        media_channels = []
        for channel in channels:
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

        if len(media_channels) == 0:
            return

        n_channels = len(media_channels)
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
            original_channel = self.map_abc_to_original_channel(channel_data['name'])

            # Определение диапазона на основе реальных данных
            if original_channel in real_media_data:
                max_historical = real_media_data[original_channel]['max']
                mean_spend = real_media_data[original_channel]['mean']
                base_max = max_historical * 2.5
            else:
                mean_spend = 25
                base_max = 100

            # Создаем предварительный диапазон для анализа
            test_range = np.linspace(0, base_max * 2, 500)

            # ABC функция
            def abc_curve(spend, a, b, c):
                if isinstance(spend, (int, float)):
                    spend = np.array([spend])

                adstocked = np.zeros_like(spend, dtype=float)
                adstocked[0] = spend[0]
                for i in range(1, len(spend)):
                    adstocked[i] = spend[i] + a * adstocked[i - 1]

                saturated = (c * adstocked) / (1 + c * adstocked + 1e-10)
                final = b * saturated
                return final

            # Анализируем кривую для определения точки обрезания
            test_response = abc_curve(test_range, A, B, C)
            derivatives = np.gradient(test_response, test_range)

            # Находим точку где производная падает до 5% от максимума
            max_derivative = np.max(derivatives[1:20])
            significant_growth_threshold = max_derivative * 0.05

            # Точка где рост становится незначительным
            low_growth_indices = np.where(derivatives < significant_growth_threshold)[0]

            if len(low_growth_indices) > 20:
                cutoff_idx = low_growth_indices[0]
                optimal_max = test_range[cutoff_idx] * 1.3
            else:
                optimal_max = base_max

            # Финальный диапазон
            final_max = min(optimal_max, base_max)

            # Создаем финальный диапазон для отображения
            spend_range = np.linspace(0, final_max, 200)
            response = abc_curve(spend_range, A, B, C)

            # Отрисовка основной кривой
            ax.plot(spend_range, response, linewidth=3, color='#2E86AB', label='Кривая отклика')
            ax.fill_between(spend_range, 0, response, alpha=0.3, color='#2E86AB')

            # Только средний уровень (желтый шарик)
            mean_response = abc_curve(np.array([mean_spend]), A, B, C)[0]
            ax.scatter([mean_spend], [mean_response], color='orange', s=100,
                       zorder=5, label=f'Средний: {mean_spend:.0f}')

            # Заголовок и подписи
            ax.set_title(f'{channel_name}\nA={A:.2f}, B={B:.2f}, C={C:.2f}',
                         fontweight='bold', pad=20)
            ax.set_xlabel('Медиа-инвестиции', fontweight='bold')
            ax.set_ylabel('Прирост продаж', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

        # Убираем лишние подграфики
        for idx in range(n_channels, len(axes)):
            axes[idx].remove()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    # Использование:
    # plot_saturation_curves(model_abc_params)

    def create_model_from_coefficients(self, train_data, forecast_data, abc_params, data_to_plot,
                                       target_col='sales', intercept=None, save_path=None):
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

        clean_data.to_csv('test.csv', index=False)

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
        clean_data['week'] = train_data['week']
        if not data_to_plot.empty:
            data_to_plot = data_to_plot[features + [target_col] + ['week']].dropna()
            clean_data = pd.concat([clean_data, data_to_plot])
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
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
        from matplotlib.ticker import FuncFormatter

        features = self.features

        # Проверяем фичи
        missing_features = [f for f in features if f not in df_data.columns]
        if missing_features:
            print(f"Отсутствующие фичи: {missing_features}")
            return

        # Подготавливаем данные
        X = df_data[features].values
        coefficients = model.coef_
        intercept = model.intercept_

        # Рассчитываем вклады
        contributions = {}

        # Константа
        contributions['Константа'] = np.full(len(df_data), intercept)

        # Вклад каждой фичи
        for i, feature in enumerate(features):
            feature_contribution = df_data[feature].values * coefficients[i]
            clean_name = feature.replace('_abc', '').replace('_', ' ').title()
            contributions[clean_name] = feature_contribution
            print(f"{clean_name}: avg={np.mean(feature_contribution):.2f}, "
                  f"min={np.min(feature_contribution):.2f}, max={np.max(feature_contribution):.2f}")

        # Временная ось
        if 'Week' in df_data.columns:
            x = pd.to_datetime(df_data['Week']) if df_data['Week'].dtype == 'object' else df_data['Week']
            x_label = 'Дата'
        elif 'week' in df_data.columns:
            x = df_data['week']
            x_label = 'Недели'
        else:
            x = range(len(df_data))
            x_label = 'Период'

        positive_contributions = {}
        negative_contributions = {}

        for name, values in contributions.items():
            if np.mean(values) >= 0:
                positive_contributions[name] = np.maximum(values, 0)  # убираем отрицательные выбросы
            else:
                negative_contributions[name] = np.minimum(values, 0)  # убираем положительные выбросы

        n_factors = len(contributions)
        colors_positive = cm.Set3(np.linspace(0, 1, len(positive_contributions)))
        colors_negative = cm.Set1(np.linspace(0, 1, len(negative_contributions)))

        fig, ax = plt.subplots(1, 1, figsize=(16, 10), dpi=100)

        if positive_contributions:
            pos_labels = list(positive_contributions.keys())
            pos_data = list(positive_contributions.values())
            ax.stackplot(x, *pos_data, labels=pos_labels, colors=colors_positive, alpha=0.8)

        if negative_contributions:
            neg_labels = list(negative_contributions.keys())
            neg_data = list(negative_contributions.values())
            ax.stackplot(x, *neg_data, labels=neg_labels, colors=colors_negative, alpha=0.8)

        if 'sales' in df_data.columns:
            actual_sales = df_data['sales'].values
            predicted_sales = sum(contributions.values())
            ax.plot(x, actual_sales, 'k-', linewidth=3, label='Фактические продажи', alpha=0.9)
            ax.plot(x, predicted_sales, 'r--', linewidth=2, label='Модельный прогноз', alpha=0.9)

            # Считаем качество
            r2 = np.corrcoef(actual_sales, predicted_sales)[0, 1] ** 2
            rmse = np.sqrt(np.mean((actual_sales - predicted_sales) ** 2))
            print(f"R²: {r2:.3f}, RMSE: {rmse:.0f}")

        ax.set_title('Декомпозиция продаж по факторам во времени\n(Положительные и отрицательные вклады)',
                     fontsize=16, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('Продажи', fontsize=12)

        ax.legend(fontsize=9, ncol=2, loc='upper left', bbox_to_anchor=(0, 1))

        # Форматирование осей
        if hasattr(x, 'dt'):  # если это даты
            ax.tick_params(axis='x', rotation=45)
        elif len(x) > 20:
            step = len(x) // 10
            plt.xticks(x[::step], rotation=45)

        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:,.0f}'))

        # Горизонтальная линия на 0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)

        # Стиль
        ax.spines["top"].set_alpha(0)
        ax.spines["right"].set_alpha(0)
        ax.spines["bottom"].set_alpha(0.3)
        ax.spines["left"].set_alpha(0.3)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        # Статистика вкладов
        print(f"\n📊 СРЕДНИЕ ВКЛАДЫ ПО ФАКТОРАМ:")
        print("=" * 50)

        total_contributions = []
        for name, values in contributions.items():
            avg_contribution = np.mean(values)
            total_contributions.append((name, avg_contribution))

        # Сортируем по убыванию вклада
        total_contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        total = 0
        for name, avg_contribution in total_contributions:
            print(f"{name:<25}: {avg_contribution:>10.0f}")
            total += avg_contribution

        print("=" * 50)
        print(f"{'ИТОГО':<25}: {total:>10.0f}")

        if 'sales' in df_data.columns:
            actual_avg = np.mean(df_data['sales'])
            print(f"{'ФАКТ (средний)':<25}: {actual_avg:>10.0f}")
            print(f"{'РАЗНИЦА':<25}: {total - actual_avg:>10.0f}")

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

    @staticmethod
    def plot_train_vs_forecast(model_data, forecast_data, predictions, target_col='sales'):
        """
        Отрисовывает факт на train + прогноз на test
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        # Получаем train данные из model_data
        train_data = model_data['transformed_data']
        print(train_data.columns)
        if 'week' in train_data.columns:
            train_dates = pd.to_datetime(train_data['week'])
            train_actual = train_data[target_col].values
        elif hasattr(train_data.index, 'to_pydatetime'):  # если индекс - даты
            train_dates = train_data.index
            train_actual = train_data[target_col].values
        else:
            print("⚠️ Не найден столбец с датами для отрисовки")
            return

        if 'week' in forecast_data.columns:
            test_dates = pd.to_datetime(forecast_data['week'])
        elif hasattr(forecast_data.index, 'to_pydatetime'):
            test_dates = forecast_data.index
        else:
            test_dates = pd.date_range(start=train_dates.max(), periods=len(predictions) + 1, freq='W')[1:]

        fig, ax = plt.subplots(figsize=(15, 8))

        ax.plot(train_dates, train_actual, 'b-', linewidth=2, label='Факт (Train)', alpha=0.8)
        ax.plot(test_dates, predictions, 'r--', linewidth=2, label='Прогноз (Test)', alpha=0.8)

        if len(train_dates) > 0 and len(test_dates) > 0:
            split_date = train_dates.max()
            ax.axvline(x=split_date, color='gray', linestyle=':', alpha=0.7,
                       label=f'Train/Test split ({split_date.strftime("%Y-%m-%d")})')

        ax.set_xlabel('Дата')
        ax.set_ylabel(target_col.title())
        ax.set_title('Факт vs Прогноз: Train + Test данные')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if len(train_dates) + len(test_dates) > 52:  # Если больше года данных
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Каждые 3 месяца
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        else:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))  # Каждые 4 недели
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

        plt.xticks(rotation=45)
        plt.tight_layout()

        print(f"Статистика прогноза:")
        print(f"Train период: {train_dates.min().strftime('%Y-%m-%d')} - {train_dates.max().strftime('%Y-%m-%d')}")
        print(f"Test период: {test_dates.min().strftime('%Y-%m-%d')} - {test_dates.max().strftime('%Y-%m-%d')}")
        print(f"Средний факт (train): {np.mean(train_actual):.2f}")
        print(f"Средний прогноз (test): {np.mean(predictions):.2f}")
        print(f"Разница средних: {np.mean(predictions) - np.mean(train_actual):.2f}")

        plt.show()

    def get_real_media_levels(self):
        """Получает РЕАЛЬНЫЕ уровни инвестиций из данных"""
        if self.train_data is None:
            return {}

        real_data = {}

        # Маппинг ABC каналов к исходным колонкам в твоих данных
        channel_mapping = {
            'federal_tv': 'federal_tv',
            'thematic_tv': 'thematic_tv',
            'regional_tv': 'regional_tv',
            'Конкурент1_ТВ_Рейтинги': 'конкурент1_тв_рейтинги',  # нормализованное название
            'Конкурент2_ТВ_Рейтинги': 'конкурент2_тв_рейтинги',
            'Конкурент3_ТВ_Рейтинги': 'конкурент3_тв_рейтинги',
            'Конкурент4_ТВ_Рейтинги': 'конкурент4_тв_рейтинги',
        }

        print("Доступные колонки в train_data:")
        print([col for col in self.train_data.columns if
               any(x in col.lower() for x in ['federal', 'thematic', 'regional', 'конкурент'])])

        for abc_name, original_name in channel_mapping.items():
            if original_name in self.train_data.columns:
                data = self.train_data[original_name].dropna()
                if len(data) > 0:
                    real_data[abc_name] = {
                        'min': float(data.min()),
                        'max': float(data.max()),
                        'mean': float(data.mean()),
                        'current': float(data.iloc[-1]),  # последнее значение
                        'q75': float(data.quantile(0.75)),  # 75-й перцентиль
                    }
                    print(
                        f"{abc_name}: min={real_data[abc_name]['min']:.1f}, max={real_data[abc_name]['max']:.1f}, current={real_data[abc_name]['current']:.1f}")
            else:
                print(f"Не найдена колонка: {original_name}")

        return real_data

    def map_abc_to_original_channel(self, abc_channel_name):
        """Маппинг ABC имени к исходному"""
        mapping = {
            'federal_tv': 'federal_tv',
            'thematic_tv': 'thematic_tv',
            'regional_tv': 'regional_tv',
            'Конкурент1_ТВ_Рейтинги': 'Конкурент1_ТВ_Рейтинги',
            'Конкурент2_ТВ_Рейтинги': 'Конкурент2_ТВ_Рейтинги',
            'Конкурент3_ТВ_Рейтинги': 'Конкурент3_ТВ_Рейтинги',
            'Конкурент4_ТВ_Рейтинги': 'Конкурент4_ТВ_Рейтинги',
        }
        return mapping.get(abc_channel_name, abc_channel_name)

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

        print(f" Выполняем прогноз для {len(X_forecast)} наблюдений...")

        try:
            predictions = model.predict(X_forecast)
        except Exception as e:
            return None

        # Результат
        result = {
            'predictions': predictions,
            # 'forecast_data': forecast_data,
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
        if target_col in forecast_data.columns and any(forecast_data[target_col]):
            actual = forecast_data[target_col].values

            mse = mean_squared_error(actual, predictions)
            r2 = r2_score(actual, predictions)
            print('Метрики на тестовых данных:')
            print(actual, predictions)
            print('rmse', np.sqrt(mse))
            print('r2', r2)
        else:
            print('kuku')
            self.plot_train_vs_forecast(model_data, forecast_data, predictions, target_col)
        # print(forecast_data, predictions)
        # self.save_forecast_results(result)
        return result


model_coef = [475.47188011388477, 655.0730197811116, 2103.7308818153565, -237.91232243849566, -1014.3196438689523, 539.3589663504629, 2816.363174461569, -4822.073524769513, -21.195168030529913, -376.63058882530095, 114.72658571909479, 283.0991711367753, -38.63932553007724]

model_features = ['federal_tv_abc', 'thematic_tv_abc', 'regional_tv_abc', 'Конкурент1_ТВ_Рейтинги_abc', 'Конкурент2_ТВ_Рейтинги_abc', 'Конкурент3_ТВ_Рейтинги_abc', 'Конкурент4_ТВ_Рейтинги_abc', 'price_ratio', 'is_spring', 'is_summer', 'is_autumn', 'is_winter', 'competitor_price_trend']

# ПРАВИЛЬНЫЕ параметры для твоей функции:
model_abc_params = {'federal_tv_A': 0.8578554509227851, 'federal_tv_B': 4.978030195753396, 'federal_tv_C': 4.744117793338168, 'thematic_tv_A': 0.8697071021624991, 'thematic_tv_B': 3.2613417490966987, 'thematic_tv_C': 0.37028284391806165, 'regional_tv_A': 0.7534743099591871, 'regional_tv_B': 0.808453454643836, 'regional_tv_C': 5.833099501032512, 'Конкурент1_ТВ_Рейтинги_A': 0.5550793311512732, 'Конкурент1_ТВ_Рейтинги_B': 1.6000699816692099, 'Конкурент1_ТВ_Рейтинги_C': 0.7746348671954331, 'Конкурент2_ТВ_Рейтинги_A': 0.6513321011751504, 'Конкурент2_ТВ_Рейтинги_B': 2.57521452253587, 'Конкурент2_ТВ_Рейтинги_C': 6.445221410745203, 'Конкурент3_ТВ_Рейтинги_A': 0.4249855292041599, 'Конкурент3_ТВ_Рейтинги_B': 0.9675091228624451, 'Конкурент3_ТВ_Рейтинги_C': 2.2449005988609114, 'Конкурент4_ТВ_Рейтинги_A': 0.7849159268740338, 'Конкурент4_ТВ_Рейтинги_B': 0.44380177532213994, 'Конкурент4_ТВ_Рейтинги_C': 5.630751549543227}

# params = pd.read_excel('model_params.xlsx')

data = pd.read_csv('data.csv', sep=';', encoding='utf-8')
data['Week'] = pd.to_datetime(data['Week'], format='%d.%m.%Y')

data = create_all_features(data.copy())
normalized_columns = list(map(lambda x: normalize_feature_name(x), data.columns))
data.columns = normalized_columns
# print(data.columns, 'qwsd')
train_end = pd.to_datetime('2012-09-24')
start_forecast = pd.to_datetime('2012-12-30')
forecast_end = pd.to_datetime('2013-12-30')

dict_params, data = add_abc(model_abc_params, data)

if train_end != start_forecast:
    data_to_plot = data[(data['week'] > train_end) & (data['week'] <= start_forecast)].copy()
else:
    data_to_plot = pd.DataFrame()

data.to_csv('data_abc.csv', index=False)

train_data = data[data['week'] <= train_end].copy()
# print(train_data['federal_tv_abc'])

forecast_data = data[(data['week'] > start_forecast) & (data['week'] <= forecast_end)].copy()

visualizer = MMMVisualizer(model_coef, model_features, model_abc_params, train_data=train_data)

model, result_predict = visualizer.create_model_from_coefficients(train_data, forecast_data, dict_params, data_to_plot)

print(result_predict)
# Строим все графики

visualizer.plot_saturation_curves()
visualizer.plot_real_contribution_over_time(train_data, model['model'])
# visualizer.plot_feature_elasticity(model['model'], train_data)

# print(model)
