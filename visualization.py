import pickle
from datetime import datetime
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from pprint import pprint


from main import create_all_features, abc_transform

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")


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
    def __init__(self, results_df):
        """
        results_df: DataFrame с колонками features, p-values, coef, A, B, C
        """
        self.results_df = results_df
        features = self.results_df['features'].tolist()
        features = list(map(lambda x: x.lower(), features))
        results_df['features'] = features

    def plot_saturation_curves(self, save_path='saturation_curves.png'):
        """
        Строит кривые насыщения для каждого медиа-канала
        """
        media_channels = self.results_df[self.results_df['C'].notna()].copy()

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

        for idx, (_, row) in enumerate(media_channels.iterrows()):
            ax = axes[idx]

            A, B, C = row['A'], row['B'], row['C']
            channel_name = row['features'].replace('_abc', '').replace('_', ' ').title()

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

        for idx in range(n_channels, len(axes)):
            axes[idx].remove()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def create_model_from_coefficients(self, transformed_data, target_col='sales',
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

        print("Создание модели из коэффициентов...")
        abc_params = {}

        for _, row in self.results_df.iterrows():
            feature_name = row['features'].lower()
            if pd.notna(row.get('A')) and pd.notna(row.get('B')) and pd.notna(row.get('C')):
                A, B, C = row['A'], row['B'], row['C']
                original_feature = feature_name.replace('_abc', '')

                if original_feature in transformed_data.columns:
                    original_values = transformed_data[original_feature].values
                    abc_transformed = abc_transform(original_values, A, B, C)
                    transformed_data[feature_name] = abc_transformed

                    abc_params[f'{original_feature}_A'] = A
                    abc_params[f'{original_feature}_B'] = B
                    abc_params[f'{original_feature}_C'] = C

                else:
                    print(f"Исходная фича не найдена: {original_feature}")
        # print('qwe', self.results_df['features'])
        # print(transformed_data.columns)
        features = self.results_df['features'].tolist()

        missing_features = [f for f in features if f not in transformed_data.columns]

        if missing_features:
            print(f"Отсутствующие признаки: {missing_features}")
            return None

        clean_data = transformed_data[features + [target_col]].dropna()
        X = clean_data[features].values
        y = clean_data[target_col].values
        coefficients = self.results_df['coef'].values

        print(f"Подготовлено данных: {len(clean_data)} наблюдений, {len(features)} признаков")

        model = LinearRegression()
        model.coef_ = coefficients
        if intercept is None:
            # intercept = mean(y) - mean(X @ coef)
            predicted_without_intercept = X @ coefficients
            intercept = np.mean(y) - np.mean(predicted_without_intercept)

        model.intercept_ = intercept
        model.n_features_in_ = len(features)
        model.feature_names_in_ = np.array(features)

        y_pred = model.predict(X)
        r2_score = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)

        print(f"📈 Качество модели:")
        print(f"   R²: {r2_score:.4f}")
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
            'r2_score': r2_score,
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
            'coefficients_df': self.results_df,
            'metadata': metadata,
            'abc_params': abc_params
        }

        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Модель сохранена: {save_path}")

        return model_data

    def plot_real_contribution_over_time(self, df_data, model, save_path='real_contribution.png'):
        """
        Реальная декомпозиция продаж по времени
        df_data: DataFrame с временными данными и фичами
        model: обученная модель
        features: список фич, которые использует модель
        """
        features = self.results_df['features'].tolist()

        # Проверяем что все фичи есть в данных
        missing_features = [f for f in features if f not in df_data.columns]
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
        print(f"{'─' * 50}")

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
        features = self.results_df['features'].tolist()
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


params = pd.read_excel('model_params.xlsx')

data = pd.read_csv('data.csv', sep=';', encoding='utf-8')
data['Week'] = pd.to_datetime(data['Week'], format='%d.%m.%Y')
data = create_all_features(data.copy())
# print(data.columns)

normalized_columns = list(map(lambda x: normalize_feature_name(x), data.columns))
data.columns = normalized_columns
# print(data.columns)
visualizer = MMMVisualizer(params)

# Строим все графики
visualizer.plot_saturation_curves()
model = visualizer.create_model_from_coefficients(data)

visualizer.plot_real_contribution_over_time(data, model['model'])
# visualizer.plot_feature_elasticity(model['model'], data)

print(model)
