import os

import pandas as pd
import numpy as np
from scipy import stats
import optuna
from sklearn.linear_model import LinearRegression, Lasso
import warnings

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings('ignore')

train_end = '2012-06-25'


def abc_transform(media_spend, A, B, C):
    """
    ABC преобразование медиа-переменной
    A: Adstock rate (0-1)
    B: Base multiplier (коэффициент масштабирования)
    C: Carryover/Saturation strength (сила насыщения)
    """
    if len(media_spend) == 0:
        return media_spend

    adstocked = np.zeros_like(media_spend, dtype=float)
    adstocked[0] = media_spend[0]
    for i in range(1, len(media_spend)):
        adstocked[i] = media_spend[i] + A * adstocked[i - 1]

    saturated = (C * adstocked) / (1 + C * adstocked + 1e-10)

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
        beta = np.linalg.inv(X_with_const.T @ X_with_const) @ X_with_const.T @ y
        y_pred = X_with_const @ beta
        residuals = y - y_pred

        mse = np.sum(residuals ** 2) / (n - k - 1)
        var_beta = mse * np.linalg.inv(X_with_const.T @ X_with_const)
        se_beta = np.sqrt(np.diag(var_beta))

        # t-статистики и p-values
        t_stats = beta / se_beta
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))

        return p_values[1:]  # Исключаем константу
    except:
        return np.ones(k)  # Возвращаем единицы при ошибке


def append_to_csv(df, filename):
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)


def extrapolate_linear_trend(data_length, intercept, slope, start_idx=0):
    indices = np.arange(start_idx, start_idx + data_length)
    return slope * indices + intercept


def apply_seasonal_pattern(data_length, seasonal_pattern):
    return np.tile(seasonal_pattern, data_length // len(seasonal_pattern) + 1)[:data_length]


def create_all_features(df):
    """
    Создает ВСЕ возможные признаки для модели с заменой NaN на 0
    """

    # === ТВ КАНАЛЫ ===

    split_date = train_end
    split_idx = df[df['Week'] <= split_date].index[-1]

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
    # изменение цены относительного предыдущего года
    df['premiumization_index'] = (
            df['Цена бренда, руб.'] / df['Цена бренда, руб.']
            .rolling(52).mean()
    ).fillna(1)

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
    # print(df['Week'])
    df['month'] = df['Week'].dt.month
    df['quarter'] = df['Week'].dt.quarter
    df['week_of_year'] = df['Week'].dt.isocalendar().week
    df['year'] = df['Week'].dt.year

    df['is_holiday_season'] = ((df['month'] == 12) | (df['month'] == 1)).astype(int)
    df['is_spring'] = ((df['month'] >= 3) & (df['month'] <= 5)).astype(int)
    df['is_summer'] = ((df['month'] >= 6) & (df['month'] <= 8)).astype(int)
    df['is_autumn'] = ((df['month'] >= 9) & (df['month'] <= 11)).astype(int)
    df['is_winter'] = ((df['month'] == 12) | (df['month'] <= 2)).astype(int)

    # competitor_price = df['Средняя цена в категории, руб.']  # замените на правильное название

    # Декомпозиция на тренд, сезонность и остатки

    train_data = df.loc[:split_idx]['Средняя цена в категории, руб.']
    decomposition = seasonal_decompose(train_data,
                                       model='additive',
                                       period=52)

    seasonal_pattern = decomposition.seasonal[:52].values

    train_trend_clean = decomposition.trend.dropna()
    train_indices = np.arange(len(train_trend_clean))
    slope, intercept, r_value, p_value, std_err = stats.linregress(train_indices, train_trend_clean)

    full_length = len(df)
    extrapolated_trend = extrapolate_linear_trend(full_length, intercept, slope)
    extrapolated_seasonal = apply_seasonal_pattern(full_length, seasonal_pattern)

    df['competitor_price_trend'] = extrapolated_trend
    df['competitor_price_seasonal'] = extrapolated_seasonal

    df['trend'] = range(len(df))

    # === ДОПОЛНИТЕЛЬНЫЕ ПРИЗНАКИ ===

    # # Взаимодействия
    # df['price_tv_interaction'] = df['price_ratio'] * df['all_tv']
    # df['price_premium_tv'] = df['price_premium'] * df['all_tv']

    # Конкурентное давление
    df['competitive_pressure'] = df['total_competitors'] / (df['all_tv'] + 1)

    # Заменяем все оставшиеся NaN на 0
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)

    # Заменяем inf на 0 (может возникнуть при делении)
    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], 0)

    print(f"Создано {len(df.columns)}")

    return df


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
                ['premiumization_index'],
                None
            ],
            'seasonal': [
                ['is_spring', 'is_summer', 'is_autumn', 'is_winter'],
                ['is_spring', 'is_summer', 'is_autumn', 'is_winter', 'is_holiday_season'],
                ['competitor_price_seasonal'],
                None
            ],
            'trends': [
                ['trend'],
                ['competitor_price_trend'],
                None
            ],
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
            'change_price': False,
            'trends': False
        }

        self.best_model = None
        self.best_ssr = float('inf')
        self.best_params = None

    def apply_abc_transformation(self, trial, data, channels):
        """
        Применяет ABC преобразование к группе каналов
        """
        transformed_features = []
        abc_params = {}

        for channel in channels:
            # Генерируем уникальные имена параметров для каждого канала
            param_prefix = f'{channel.replace(" ", "_").replace(",", "").replace(".", "")}'

            abc_params[f'{param_prefix}_A'] = trial.suggest_float(f'{param_prefix}_A', 0.0, 0.9)
            abc_params[f'{param_prefix}_B'] = trial.suggest_float(f'{param_prefix}_B', 0.01, 3.0)
            abc_params[f'{param_prefix}_C'] = trial.suggest_float(f'{param_prefix}_C', 0.001, 5.0)
            print(channel, 'channel')
            A = abc_params[f'{param_prefix}_A']
            B = abc_params[f'{param_prefix}_B']
            C = abc_params[f'{param_prefix}_C']

            # Применяем ABC преобразование
            transformed = abc_transform(data[channel].values, A, B, C)
            transformed_feature_name = f'{param_prefix}_abc'

            data[transformed_feature_name] = transformed
            transformed_features.append(transformed_feature_name)
            print(transformed_feature_name, 'transformed_feature_name')

        return transformed_features, abc_params

    def objective(self, trial):
        """
        Упрощенная целевая функция без дублирования кода
        """

        data = self.train_data.copy()
        selected_features = []
        all_abc_params = {}
        # df1 =
        split_idx = data[(data['Week'] == train_end)].index[0] + 1

        # === ВЫБИРАЕМ КОНФИГУРАЦИИ ===

        selected_configs = {}
        for group_name, configs in self.feature_configs.items():
            config_idx = trial.suggest_int(f'{group_name}_config', 0, len(configs) - 1)
            selected_configs[group_name] = configs[config_idx]

        # === ОБРАБАТЫВАЕМ КАЖДУЮ ГРУППУ ===
        abc_params = None
        print(selected_configs.items(), 'selected_configs.items()')
        for group_name, selected_channels in selected_configs.items():
            if selected_channels is None:
                continue

            # Проверяем нужны ли ABC преобразования для этой группы
            if self.abc_groups[group_name]:
                # Применяем ABC преобразование
                transformed_features, abc_params = self.apply_abc_transformation(
                    trial, data, selected_channels
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
        # split_idx = int(len(data_clean) * 0.8)

        X_train = data_clean[selected_features][data_clean.index < split_idx].values
        y_train = data_clean[self.target_col][data_clean.index < split_idx].values
        X_test = data_clean[selected_features][data_clean.index >= split_idx].values
        y_test = data_clean[self.target_col][data_clean.index >= split_idx].values

        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        try:
            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            base_ssr = mean_squared_error(y_test, y_pred)

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
            p_values = calculate_p_values(X_train, y_train)

            insignificant_penalty = np.sum(p_values > 0.1) * 0.05
            # complexity_penalty = max(0, (len(selected_features) - 3) * 0.01)
            data_clean.to_csv('data_clean_main.csv')
            if abc_params:
                self._print_model_statistics(model, selected_features, data_clean, all_abc_params, split_idx)

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

        print(f"\nЛУЧШАЯ КОНФИГУРАЦИЯ:")
        print(f"{'─' * 60}")
        for group_name, selected_features in best_config.items():
            abc_marker = " (с ABC)" if self.abc_groups[group_name] and selected_features else ""
            if selected_features is not None:
                print(f"  {group_name.upper()}{abc_marker}: {selected_features}")
            else:
                print(f"  {group_name.upper()}: НЕ ИСПОЛЬЗУЕТСЯ")

        print(f"\nSSR: {study.best_value:.2f}")

        # Выводим ABC параметры
        print(f"\nABC ПАРАМЕТРЫ:")
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

    def _decode_configuration(self, params):
        """
        Декодирует конфигурацию из параметров оптимизации
        """
        selected_configs = {}
        for group_name, configs in self.feature_configs.items():
            config_idx = params[f'{group_name}_config']
            selected_configs[group_name] = configs[config_idx]
        return selected_configs

    def _print_model_statistics(self, model, selected_features, data_clean, params, split_idx):
        """
        Выводит статистики модели и коэффициенты
        """
        X_train = data_clean[selected_features][data_clean.index < split_idx].values
        y_train = data_clean[self.target_col][data_clean.index < split_idx].values
        X_test = data_clean[selected_features][data_clean.index >= split_idx].values
        y_test = data_clean[self.target_col][data_clean.index >= split_idx].values

        # abc_params = {k: v for k, v in params.items() if k.endswith('_A') or k.endswith('_B') or k.endswith('_C')}
        y_pred = model.predict(X_test)
        ssr = np.sum((y_test - y_pred) ** 2)
        r2 = model.score(X_test, y_test)
        rmse = np.sqrt(ssr / len(y_test))
        p_values = calculate_p_values(X_train, y_train)
        df_statistic_model = pd.DataFrame()
        df_statistic_model['features'] = [list(selected_features)]
        df_statistic_model['p-values'] = [list(p_values)]
        df_statistic_model['coef'] = [list(model.coef_)]
        df_statistic_model['r2'] = r2
        df_statistic_model['rmse'] = rmse
        df_statistic_model['params_abc'] = str(params)
        append_to_csv(df_statistic_model, 'statistic_model.csv')


def main():
    df = pd.read_csv('data.csv', sep=';', encoding='utf-8')
    df['Week'] = pd.to_datetime(df['Week'], format='%d.%m.%Y')
    df = df.sort_values('Week').reset_index(drop=True)
    forecast_end = pd.to_datetime('2012-12-30')
    df = df[df['Week'] <= forecast_end]
    # train_data = df[df['Week'] <= train_end].copy()
    # forecast_data = df[(df['Week'] > train_end) & (df['Week'] <= forecast_end)].copy()

    # Оптимизация
    optimizer = ABCOptimizer(df)
    best_params, best_ssr = optimizer.optimize(n_trials=300)


if __name__ == "__main__":
    main()
