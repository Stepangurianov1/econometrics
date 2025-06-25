import pandas as pd
import numpy as np
from scipy import stats
import optuna
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')


# ABC преобразование
def abc_transform(media_spend, A, B, C):
    """
    ABC преобразование медиа-переменной
    A: Adstock rate (0-1)
    B: Base multiplier (коэффициент масштабирования)
    C: Carryover/Saturation strength (сила насыщения)
    """
    if len(media_spend) == 0:
        return media_spend

    # Шаг 1: Adstock преобразование
    adstocked = np.zeros_like(media_spend, dtype=float)
    adstocked[0] = media_spend[0]
    for i in range(1, len(media_spend)):
        adstocked[i] = media_spend[i] + A * adstocked[i - 1]

    # Шаг 2: Saturation curve (Hill transformation)
    # Избегаем деление на ноль
    saturated = (C * adstocked) / (1 + C * adstocked + 1e-10)

    # Шаг 3: Base scaling
    final = B * saturated

    return final


# Функция расчета p_value (оставляем как была)
def calculate_p_values(X, y):
    """
    Рассчитывает p-values для коэффициентов линейной регрессии
    """
    n = X.shape[0]
    k = X.shape[1]

    # Добавляем константу
    X_with_const = np.column_stack([np.ones(n), X])

    try:
        # Вычисляем коэффициенты
        beta = np.linalg.inv(X_with_const.T @ X_with_const) @ X_with_const.T @ y

        # Предсказания и остатки
        y_pred = X_with_const @ beta
        residuals = y - y_pred

        # Стандартная ошибка
        mse = np.sum(residuals ** 2) / (n - k - 1)
        var_beta = mse * np.linalg.inv(X_with_const.T @ X_with_const)
        se_beta = np.sqrt(np.diag(var_beta))

        # t-статистики и p-values
        t_stats = beta / se_beta
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))

        return p_values[1:]  # Исключаем константу
    except:
        return np.ones(k)  # Возвращаем единицы при ошибке


# Создание признаков (оставляем как было)
def create_tv_features(df):
    """
    Создает различные группировки ТВ каналов
    """
    # Федеральные каналы
    federal_channels = ['Первый Канал, ТВ Рейтинги', 'НТВ, ТВ Рейтинги', 'Пятый Канал, ТВ Рейтинги']
    df['federal_tv'] = df[federal_channels].sum(axis=1)

    # Тематические каналы
    thematic_channels = ['Домашний, ТВ Рейтинги', 'ТВ-3, ТВ Рейтинги',
                         'Рен ТВ, ТВ Рейтинги', 'Звезда, ТВ Рейтинги']
    df['thematic_tv'] = df[thematic_channels].sum(axis=1)

    # Региональные каналы
    regional_channels = ['ТВ Центр, ТВ Рейтинги', 'Нишевые каналы, ТВ Рейтинги']
    df['regional_tv'] = df[regional_channels].sum(axis=1)

    # Все каналы
    all_tv_channels = federal_channels + thematic_channels + regional_channels + ['Россия 1, ТВ Рейтинги']
    df['all_tv'] = df[all_tv_channels].sum(axis=1)

    # Конкуренты
    competitor_channels = ['Конкурент1, ТВ Рейтинги', 'Конкурент2, ТВ Рейтинги',
                           'Конкурент3, ТВ Рейтинги', 'Конкурент4, ТВ Рейтинги']
    df['total_competitors'] = df[competitor_channels].sum(axis=1)

    return df, all_tv_channels, competitor_channels


def create_all_features(df):
    """
    Создает ВСЕ возможные признаки для модели с заменой NaN на 0
    """

    # === ТВ КАНАЛЫ ===

    federal_channels = ['Первый Канал, ТВ Рейтинги', 'НТВ, ТВ Рейтинги', 'Пятый Канал, ТВ Рейтинги']
    df['federal_tv'] = df[federal_channels].sum(axis=1)

    thematic_channels = ['Домашний, ТВ Рейтинги', 'ТВ-3, ТВ Рейтинги',
                         'Рен ТВ, ТВ Рейтинги', 'Звезда, ТВ Рейтинги']
    df['thematic_tv'] = df[thematic_channels].sum(axis=1)

    regional_channels = ['ТВ Центр, ТВ Рейтинги', 'Нишевые каналы, ТВ Рейтинги']
    df['regional_tv'] = df[regional_channels].sum(axis=1)

    all_tv_channels = federal_channels + thematic_channels + regional_channels + ['Россия 1, ТВ Рейтинги']
    df['all_tv'] = df[all_tv_channels].sum(axis=1)

    competitor_channels = ['Конкурент1, ТВ Рейтинги', 'Конкурент2, ТВ Рейтинги',
                           'Конкурент3, ТВ Рейтинги', 'Конкурент4, ТВ Рейтинги']
    df['total_competitors'] = df[competitor_channels].sum(axis=1)

    # === ЦЕНОВЫЕ ПРИЗНАКИ ===

    df['price_ratio'] = df['Цена бренда, руб.'] / df['Средняя цена в категории, руб.']
    df['price_premium'] = df['Цена бренда, руб.'] - df['Средняя цена в категории, руб.']
    df['log_price_ratio'] = np.log(df['Цена бренда, руб.'] / df['Средняя цена в категории, руб.'])
    df['price'] = df['Цена бренда, руб.']
    df['price_ratio_lag1'] = df['price_ratio'].shift(1)
    df['price_ratio_lag2'] = df['price_ratio'].shift(2)

    # Изменения цен
    df['category_price_change'] = df['Средняя цена в категории, руб.'].pct_change()
    df['avg_price_category'] = df['Средняя цена в категории, руб.']

    # Ценовое позиционирование
    df['is_premium'] = (df['price_ratio'] > 1.1).astype(int)
    df['is_discount'] = (df['price_ratio'] < 0.9).astype(int)
    df['is_parity'] = ((df['price_ratio'] >= 0.9) & (df['price_ratio'] <= 1.1)).astype(int)

    # Скользящие средние
    df['price_ratio_ma4'] = df['price_ratio'].rolling(window=4).mean().fillna(0)
    df['price_ratio_ma12'] = df['price_ratio'].rolling(window=12).mean().fillna(0)

    # Волатильность
    df['price_volatility'] = df['price_ratio'].rolling(window=8).std().fillna(0)

    # === СЕЗОННЫЕ ПРИЗНАКИ ===

    df['month'] = df['Week'].dt.month
    df['quarter'] = df['Week'].dt.quarter
    df['week_of_year'] = df['Week'].dt.isocalendar().week
    df['year'] = df['Week'].dt.year

    df['is_holiday_season'] = ((df['month'] == 12) | (df['month'] == 1)).astype(int)
    df['is_spring'] = ((df['month'] >= 3) & (df['month'] <= 5)).astype(int)
    df['is_summer'] = ((df['month'] >= 6) & (df['month'] <= 8)).astype(int)
    df['is_autumn'] = ((df['month'] >= 9) & (df['month'] <= 11)).astype(int)
    df['is_winter'] = ((df['month'] == 12) | (df['month'] <= 2)).astype(int)

    df['trend'] = range(len(df))

    # Циклические признаки
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

    # === ДОПОЛНИТЕЛЬНЫЕ ПРИЗНАКИ ===

    # # Взаимодействия
    # df['price_tv_interaction'] = df['price_ratio'] * df['all_tv']
    # df['price_premium_tv'] = df['price_premium'] * df['all_tv']

    # Конкурентное давление
    df['competitive_pressure'] = df['total_competitors'] / (df['all_tv'] + 1)

    # Медиа-микс
    df['tv_press_ratio'] = df['all_tv'] / (df['Реклама в прессе, руб.'] + 1)
    df['total_media'] = df['all_tv'] + df['Реклама в прессе, руб.']


    # === ФИНАЛЬНАЯ ОЧИСТКА NaN ===

    # Заменяем все оставшиеся NaN на 0
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)

    # Заменяем inf на 0 (может возникнуть при делении)
    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], 0)

    print(f"Создано {len(df.columns)} признаков, все NaN заменены на 0")

    return df


# Упрощенный ABCOptimizer
class ABCOptimizer:
    def __init__(self, train_data, target_col='Sales'):
        self.train_data = create_all_features(train_data.copy())
        self.target_col = target_col

        # Конфигурация возможных комбинаций признаков
        self.feature_configs = {
            'tv': [
                ['federal_tv', 'thematic_tv', 'regional_tv'],
                ['all_tv'],
                ['federal_tv'],
                ['thematic_tv'],
                ['regional_tv'],
                ['federal_tv', 'thematic_tv'],
            ],
            'competitors': [
                ['total_competitors'],
                None
            ],
            'press': [
                ['Реклама в прессе, руб.'],
                None
            ],
            'price': [
                ['price_ratio'],
                ['price_premium'],
                ['is_premium'],
                ['is_discount'],
                ['price'],
                ['price_ratio_lag1'],
                ['price_ratio_lag2'],
                None
            ],
            'seasonal': [
                ['trend'],
                ['is_holiday_season'],
                ['is_spring', 'is_summer', 'is_autumn'],
                ['trend', 'is_holiday_season'],
                ['trend', 'is_spring', 'is_summer', 'is_autumn'],
                None
            ],
            # 'interactions': [
            #     ['price_tv_interaction'],
            #     ['competitive_pressure'],
            #     None
            # ],
            'change_price': [
                ['category_price_change'],
                ['avg_price_category'],
                None
            ]
        }

        # Конфигурация ABC коэффициентов - какие группы нуждаются в ABC преобразовании
        self.abc_groups = {
            'tv': True,  # ТВ каналы нуждаются в ABC
            'competitors': True,  # Конкуренты нуждаются в ABC
            'press': True,  # Пресса нуждается в ABC
            'price': False,  # Ценовые признаки БЕЗ ABC
            'seasonal': False,  # Сезонные признаки БЕЗ ABC
            'interactions': False,  # Взаимодействия БЕЗ ABC
            'autoregressive': False,  # Авторегрессивные БЕЗ ABC
            'change_price': False
        }

        self.best_model = None
        self.best_ssr = float('inf')
        self.best_params = None

    def apply_abc_transformation(self, trial, data, group_name, channels):
        """
        Применяет ABC преобразование к группе каналов
        """
        transformed_features = []
        abc_params = {}

        for channel in channels:
            # Генерируем уникальные имена параметров для каждого канала
            param_prefix = f'{channel.replace(" ", "_").replace(",", "").replace(".", "")}'

            abc_params[f'{param_prefix}_A'] = trial.suggest_float(f'{param_prefix}_A', 0.0, 0.9)
            abc_params[f'{param_prefix}_B'] = trial.suggest_float(f'{param_prefix}_B', 0.01, 10.0)
            abc_params[f'{param_prefix}_C'] = trial.suggest_float(f'{param_prefix}_C', 0.001, 5.0)

            A = abc_params[f'{param_prefix}_A']
            B = abc_params[f'{param_prefix}_B']
            C = abc_params[f'{param_prefix}_C']

            # Применяем ABC преобразование
            transformed = abc_transform(data[channel].values, A, B, C)
            transformed_feature_name = f'{param_prefix}_abc'
            data[transformed_feature_name] = transformed
            transformed_features.append(transformed_feature_name)

        return transformed_features, abc_params

    def objective(self, trial):
        """
        Упрощенная целевая функция без дублирования кода
        """

        data = self.train_data.copy()
        selected_features = []
        all_abc_params = {}

        # === ВЫБИРАЕМ КОНФИГУРАЦИИ ===

        selected_configs = {}
        for group_name, configs in self.feature_configs.items():
            config_idx = trial.suggest_int(f'{group_name}_config', 0, len(configs) - 1)
            selected_configs[group_name] = configs[config_idx]

        # === ОБРАБАТЫВАЕМ КАЖДУЮ ГРУППУ ===

        for group_name, selected_channels in selected_configs.items():
            if selected_channels is None:
                continue

            # Проверяем нужны ли ABC преобразования для этой группы
            if self.abc_groups[group_name]:
                # Применяем ABC преобразование
                transformed_features, abc_params = self.apply_abc_transformation(
                    trial, data, group_name, selected_channels
                )
                selected_features.extend(transformed_features)
                all_abc_params.update(abc_params)
            else:
                # Просто добавляем признаки как есть
                selected_features.extend(selected_channels)

        # === ПРОВЕРКИ И ОБУЧЕНИЕ ===

        if len(selected_features) < 1:
            return float('inf')

        # Проверяем наличие признаков
        missing_features = [f for f in selected_features if f not in data.columns]
        if missing_features:
            return float('inf')

        data_clean = data[selected_features + [self.target_col]].dropna()

        if len(data_clean) < 20:
            return float('inf')

        X = data_clean[selected_features].values
        y = data_clean[self.target_col].values

        # Проверка на константность
        feature_vars = np.var(X, axis=0)
        if np.any(feature_vars < 1e-10):
            return float('inf')

        try:
            model = LinearRegression()
            model.fit(X, y)

            y_pred = model.predict(X)
            base_ssr = np.sum((y - y_pred) ** 2)

            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                return float('inf')

            media_penalty = 0
            for i, feature in enumerate(selected_features):
                if 'abc' in feature:  # Это медиа-признак
                    if 'competitors' in feature:
                        if model.coef_[i] > 0:
                            # Большой штраф за отрицательный медиа-коэффициент
                            media_penalty += abs(model.coef_[i]) * 1000
                    else:
                        if model.coef_[i] < 0:
                            media_penalty += abs(model.coef_[i]) * 1000

            # Остальные штрафы
            p_values = calculate_p_values(X, y)
            insignificant_penalty = np.sum(p_values > 0.1) * 0.05
            # complexity_penalty = max(0, (len(selected_features) - 3) * 0.01)

            total_penalty = media_penalty + insignificant_penalty
            penalized_ssr = base_ssr * (1 + total_penalty)

            return penalized_ssr

        except:
            return float('inf')

    def optimize(self, n_trials=300):
        """
        Запуск оптимизации с декодированием результатов
        """
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)

        self.best_params = study.best_params
        self.best_ssr = study.best_value

        # Декодируем лучшую конфигурацию
        best_config = {}
        for group_name, configs in self.feature_configs.items():
            config_idx = study.best_params[f'{group_name}_config']
            best_config[group_name] = configs[config_idx]

        print(f"\n🎯 ЛУЧШАЯ КОНФИГУРАЦИЯ:")
        print(f"{'─' * 60}")
        for group_name, selected_features in best_config.items():
            abc_marker = " (с ABC)" if self.abc_groups[group_name] and selected_features else ""
            if selected_features is not None:
                print(f"  {group_name.upper()}{abc_marker}: {selected_features}")
            else:
                print(f"  {group_name.upper()}: НЕ ИСПОЛЬЗУЕТСЯ")

        print(f"\nSSR: {study.best_value:.2f}")

        # Выводим ABC параметры
        print(f"\n📊 ABC ПАРАМЕТРЫ:")
        print(f"{'─' * 60}")
        abc_params = {k: v for k, v in study.best_params.items() if
                      k.endswith('_A') or k.endswith('_B') or k.endswith('_C')}

        # Группируем по каналам
        channels = set()
        for param_name in abc_params.keys():
            channel = param_name.rsplit('_', 1)[0]  # Убираем _A, _B, _C
            channels.add(channel)

        for channel in sorted(channels):
            A = abc_params.get(f'{channel}_A', 'N/A')
            B = abc_params.get(f'{channel}_B', 'N/A')
            C = abc_params.get(f'{channel}_C', 'N/A')
            print(f"  {channel}: A={A:.3f}, B={B:.3f}, C={C:.3f}")

        return study.best_params, study.best_value

    def build_final_model(self, params):
        """
        Строит финальную модель с лучшими параметрами (новая архитектура)
        """
        data = self.train_data.copy()
        selected_features = []

        # === ДЕКОДИРУЕМ КОНФИГУРАЦИЮ ===

        selected_configs = {}
        for group_name, configs in self.feature_configs.items():
            config_idx = params[f'{group_name}_config']
            selected_configs[group_name] = configs[config_idx]

        # === ПРИМЕНЯЕМ ПРЕОБРАЗОВАНИЯ ===

        for group_name, selected_channels in selected_configs.items():
            if selected_channels is None:
                continue

            # Проверяем нужны ли ABC преобразования
            if self.abc_groups[group_name]:
                # Применяем ABC с сохраненными параметрами
                for channel in selected_channels:
                    param_prefix = f'{channel.replace(" ", "_").replace(",", "").replace(".", "")}'

                    A = params[f'{param_prefix}_A']
                    B = params[f'{param_prefix}_B']
                    C = params[f'{param_prefix}_C']

                    transformed = abc_transform(data[channel].values, A, B, C)
                    transformed_feature_name = f'{param_prefix}_abc'
                    data[transformed_feature_name] = transformed
                    selected_features.append(transformed_feature_name)
            else:
                # Добавляем признаки как есть
                selected_features.extend(selected_channels)

        # === ПОДГОТОВКА ДАННЫХ ===

        data_clean = data[selected_features + [self.target_col]].dropna()

        X = data_clean[selected_features].values
        y = data_clean[self.target_col].values

        # === ОБУЧЕНИЕ МОДЕЛИ ===

        model = LinearRegression()
        model.fit(X, y)

        # === РАСЧЕТ СТАТИСТИК ===

        y_pred = model.predict(X)
        ssr = np.sum((y - y_pred) ** 2)
        r2 = model.score(X, y)
        mse = ssr / len(y)
        rmse = np.sqrt(mse)

        # P-values
        p_values = calculate_p_values(X, y)

        # === КРАСИВЫЙ ОТЧЕТ ===

        print(f"\n{'=' * 80}")
        print(f"ФИНАЛЬНАЯ МОДЕЛЬ - РЕЗУЛЬТАТЫ")
        print(f"{'=' * 80}")

        print(f"\nКОНФИГУРАЦИЯ МОДЕЛИ:")
        print(f"{'─' * 60}")
        for group_name, selected_channels in selected_configs.items():
            abc_marker = " (с ABC)" if self.abc_groups[group_name] and selected_channels else ""
            if selected_channels is not None:
                print(f"  {group_name.upper()}{abc_marker}: {selected_channels}")
            else:
                print(f"  {group_name.upper()}: НЕ ИСПОЛЬЗУЕТСЯ")

        print(f"\n📈 МЕТРИКИ КАЧЕСТВА:")
        print(f"{'─' * 50}")
        print(f"  SSR (сумма квадратов остатков): {ssr:,.0f}")
        print(f"  R² (коэффициент детерминации):  {r2:.4f}")
        print(f"  RMSE (среднеквадратичная ошибка): {rmse:,.0f}")
        print(f"  Количество признаков:           {len(selected_features)}")
        print(f"  Количество наблюдений:          {len(data_clean)}")

        print(f"\nКОЭФФИЦИЕНТЫ И ЗНАЧИМОСТЬ:")
        print(f"{'─' * 80}")
        print(f"{'Признак':<35} {'Коэффициент':<15} {'P-value':<12} {'Значимость':<12}")
        print(f"{'─' * 80}")

        for i, (feature, p_val) in enumerate(zip(selected_features, p_values)):
            coef = model.coef_[i]

            if p_val < 0.001:
                significance = "***"
                significance_text = "Высокая"
            elif p_val < 0.01:
                significance = "**"
                significance_text = "Средняя"
            elif p_val < 0.05:
                significance = "*"
                significance_text = "Низкая"
            elif p_val < 0.1:
                significance = "."
                significance_text = "Слабая"
            else:
                significance = " "
                significance_text = "Не значим"

            print(f"{feature:<35} {coef:>14.4f} {p_val:>11.4f} {significance_text:<12}")

        print(f"\n📝 ABC ПАРАМЕТРЫ:")
        print(f"{'─' * 80}")

        # Группируем ABC параметры по каналам
        abc_params = {k: v for k, v in params.items() if k.endswith('_A') or k.endswith('_B') or k.endswith('_C')}
        channels = set()
        for param_name in abc_params.keys():
            channel = param_name.rsplit('_', 1)[0]
            channels.add(channel)

        for channel in sorted(channels):
            A = abc_params.get(f'{channel}_A', 'N/A')
            B = abc_params.get(f'{channel}_B', 'N/A')
            C = abc_params.get(f'{channel}_C', 'N/A')

            # Интерпретация
            if A != 'N/A':
                if A < 0.3:
                    adstock_text = "Быстрое затухание"
                elif A < 0.6:
                    adstock_text = "Умеренное затухание"
                else:
                    adstock_text = "Долгое затухание"

                if C < 1.0:
                    saturation_text = "Низкое насыщение"
                elif C < 3.0:
                    saturation_text = "Умеренное насыщение"
                else:
                    saturation_text = "Высокое насыщение"

                print(f"\n  {channel.replace('_', ' ').title()}:")
                print(f"    Adstock (A): {A:.3f} - {adstock_text}")
                print(f"    Base (B):    {B:.3f} - Множитель эффекта")
                print(f"    Carryover (C): {C:.3f} - {saturation_text}")

        print(f"\n РЕКОМЕНДАЦИИ:")
        print(f"{'─' * 50}")

        # Анализ значимых коэффициентов
        significant_features = [(selected_features[i], model.coef_[i], p_values[i])
                                for i in range(len(selected_features)) if p_values[i] < 0.1]

        if any('abc' in feat[0] for feat in significant_features):
            print(f"  Найдены значимые медиа-эффекты")

            # Лучший медиа канал
            media_effects = [(feat, coef) for feat, coef, p in significant_features if 'abc' in feat]
            if media_effects:
                best_media = max(media_effects, key=lambda x: abs(x[1]))
                print(f" Наиболее эффективный канал: {best_media[0].replace('_abc', '').replace('_', ' ').title()}")

        if any('price' in feat[0] for feat in significant_features):
            print(f" Обнаружено значимое влияние ценовых факторов")

        if any('sales_lag' in feat[0] or 'sales_ma' in feat[0] for feat in significant_features):
            print(f" Найдены авторегрессивные эффекты (инерция продаж)")

        print(f"\n{'=' * 80}")

        return model, selected_features, data_clean

    def make_forecast(self, model, features, forecast_data, best_params):
        """
        Делает прогноз на новых данных (новая архитектура)
        """

        # === ПОДГОТОВКА ПОЛНОГО ДАТАСЕТА ===

        # Объединяем тренировочные и прогнозные данные для правильных лагов и ABC
        full_data = pd.concat([self.train_data, forecast_data], ignore_index=True)

        # Применяем создание всех признаков к полному датасету
        full_prepared = create_all_features(full_data.copy())
        data = full_prepared.copy()

        # === ДЕКОДИРУЕМ КОНФИГУРАЦИЮ ===

        selected_configs = {}
        for group_name, configs in self.feature_configs.items():
            config_idx = best_params[f'{group_name}_config']
            selected_configs[group_name] = configs[config_idx]

        # === ПРИМЕНЯЕМ ТЕ ЖЕ ПРЕОБРАЗОВАНИЯ ===

        forecast_features = []

        for group_name, selected_channels in selected_configs.items():
            if selected_channels is None:
                continue

            if self.abc_groups[group_name]:
                # Применяем ABC с теми же параметрами
                for channel in selected_channels:
                    param_prefix = f'{channel.replace(" ", "_").replace(",", "").replace(".", "")}'

                    A = best_params[f'{param_prefix}_A']
                    B = best_params[f'{param_prefix}_B']
                    C = best_params[f'{param_prefix}_C']

                    transformed = abc_transform(data[channel].values, A, B, C)
                    transformed_feature_name = f'{param_prefix}_abc'
                    data[transformed_feature_name] = transformed
                    forecast_features.append(transformed_feature_name)
            else:
                # Добавляем признаки как есть
                forecast_features.extend(selected_channels)

        # === ВЫДЕЛЯЕМ ПРОГНОЗНЫЙ ПЕРИОД ===

        train_len = len(self.train_data)
        forecast_portion = data.iloc[train_len:].copy()

        # Проверяем что все нужные признаки есть
        if not all(feat in forecast_portion.columns for feat in forecast_features):
            missing = [feat for feat in forecast_features if feat not in forecast_portion.columns]
            print(f"Отсутствующие признаки: {missing}")
            return None, None

        # === ПРОГНОЗИРОВАНИЕ ===

        # Подготавливаем данные (все NaN уже заменены на 0 в create_all_features)
        X_forecast = forecast_portion[forecast_features].values

        # Проверяем на NaN (на всякий случай)
        if np.any(np.isnan(X_forecast)):
            print("Найдены NaN в прогнозных данных, заменяем на 0")
            X_forecast = np.nan_to_num(X_forecast, 0)

        # Делаем прогноз
        y_forecast = model.predict(X_forecast)

        print(f"Прогноз выполнен для {len(y_forecast)} наблюдений")

        return y_forecast, forecast_portion


def main():
    df = pd.read_csv('data.csv', sep=';', encoding='utf-8')
    df['Week'] = pd.to_datetime(df['Week'], format='%d.%m.%Y')
    df = df.sort_values('Week').reset_index(drop=True)

    # Разделение данных
    train_end = pd.to_datetime('2012-06-30')
    forecast_end = pd.to_datetime('2012-12-30')

    train_data = df[df['Week'] <= train_end].copy()
    forecast_data = df[(df['Week'] > train_end) & (df['Week'] <= forecast_end)].copy()

    print(f"Обучающая выборка: {len(train_data)} недель")
    print(f"Прогнозный период: {len(forecast_data)} недель")

    # Оптимизация
    optimizer = ABCOptimizer(train_data)
    best_params, best_ssr = optimizer.optimize(n_trials=2000)

    # Построение финальной модели
    final_model, features, clean_data = optimizer.build_final_model(best_params)

    # Прогнозирование
    forecast_result = optimizer.make_forecast(final_model, features, forecast_data, best_params)

    if forecast_result[0] is not None:
        forecast_predictions, forecast_portion = forecast_result

        # Оценка качества прогноза
        actual_forecast = forecast_data['Sales'].values[:len(forecast_predictions)]

        forecast_ssr = np.sum((actual_forecast - forecast_predictions) ** 2)

        print(f"\nКАЧЕСТВО ПРОГНОЗА:")
        print(f"{'─' * 50}")
        print(f"  SSR на прогнозном периоде: {forecast_ssr:,.0f}")
        print(f"  RMSE прогноза: {np.sqrt(forecast_ssr / len(actual_forecast)):,.0f}")

        # Сохранение результатов
        results_df = pd.DataFrame({
            'Week': forecast_data['Week'].iloc[:len(forecast_predictions)],
            'Actual': actual_forecast,
            'Predicted': forecast_predictions,
            'Residual': actual_forecast - forecast_predictions
        })

        results_df.to_csv('forecast_results.csv', index=False)
        print(f"езультаты сохранены в forecast_results.csv")
    else:
        print("Прогнозирование не удалось")


if __name__ == "__main__":
    main()