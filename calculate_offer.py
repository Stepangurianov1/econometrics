import pandas as pd
import inspect


class CalculateOffer:
    def __init__(self, card_id):
        # твой существующий код...
        self.plu_list = """3676974 3677543 3993218 4204104 4056934 4300439 3276702 3276703
                           3358963 4176574 3955551 3955552 4204103 4204105 4205847 3695795
                           3695892 4172089 4245232 3214122 2143824 3603069 4268315 3686848
                           3686847 3643212 3981707 3981711 3970742 3502796 3502794 3502795 
                           3502797 4056933 4052669 3995765 4167581 4167580 4077536 4145485
                           4195078 4348524 4248919 4369811 3622920""".split(' ')
        self.awards = 'award_type_rto'
        self.auditorium = 'loyal_calc'
        self.rules = self._init_rules()

    def _init_rules(self):
        return {
            # avg_qty - среднее потребление в штуках;
            # avg_rub - среднее потребление в рублях
            # avg_price - средняя стоимость

            # Продовольственные товары <30% >60 руб.
            ('продовольственные', '<30%', '>60'): {
                'mechanics': ['award_type_rto', 'award_type_plu_count', 'award_type_cashback'],
                'loyal_threshold': {
                    'award_type_rto': lambda avg_rub: round(avg_rub * 1.2 / 10) * 10,
                    'award_type_plu_count': lambda avg_qty: round(avg_qty * 1.2),
                    'award_type_cashback': lambda avg_rub: round(avg_rub * 1.2)
                },
                'non_loyal_threshold': {
                    'award_type_rto': lambda avg_rub: round(avg_rub / 10) * 10,
                    'award_type_plu_count': lambda: 1,
                    'award_type_cashback': lambda: 0
                },

                'loyal_offer': {
                    'award_type_rto': lambda avg_price: round((avg_price * 0.25) / 50) * 500,
                    'award_type_plu_count': lambda avg_price: round((avg_price * 0.25) / 50) * 500,
                    'award_type_cashback': lambda: 0.25
                },
                'non_loyal_offer': {
                    'award_type_rto': lambda avg_price: round((avg_price * 0.3) / 50) * 500,
                    'award_type_plu_count': lambda avg_price: round((avg_price * 0.3) / 50) * 500,
                    'award_type_cashback': lambda: 0.30
                }
            },

            # Продовольственные товары <=30% <=60 руб.
            ('продовольственные', '<=30%', '<=60'): {
                'mechanics': ['award_type_rto', 'award_type_cashback', 'award_type_plu_count'],
                'loyal_threshold': {
                    'award_type_rto': lambda avg_rub: round(avg_rub * 1.2 / 10) * 10,
                    'award_type_cashback': lambda avg_rub: round(avg_rub * 1.2 / 10) * 10,
                    'award_type_plu_count': lambda avg_qty: max(round(avg_qty * 1.2), 2)
                },
                'non_loyal_threshold': {
                    'award_type_rto': lambda avg_price: round(avg_price * 2 / 10) * 10,
                    'award_type_cashback': lambda avg_price: round(avg_price * 2 / 10) * 10,
                    'award_type_plu_count': 2
                },
                'loyal_offer': {
                    'award_type_rto': lambda avg_price: round((avg_price * 0.25 * len(self.plu_list)) * 10 / 50) * 50,
                    'award_type_cashback': lambda: 0.25,
                    'award_type_plu_count': lambda threshold: round((threshold * 0.25) * 10 / 50) * 50
                },
                'non_loyal_offer': {
                    'award_type_rto': lambda avg_price: round((avg_price * 0.30 * 2) * 10 / 50) * 50,
                    'award_type_cashback': lambda: 0.30,
                    'award_type_plu_count': lambda avg_price: round((avg_price * 0.30 * 2) * 10 / 50) * 50,
                }
            },
            # Продовольственные товары >30%
            ('продовольственные', '>30%', 'any'): {
                'mechanics': ['award_type_plu_count', 'award_type_cashback'],
                'loyal_threshold': {
                    'award_type_rto': lambda avg_rub: round(avg_rub * 1.2 / 10) * 10,
                    'award_type_cashback': lambda avg_rub: round(avg_rub * 1.2 / 10) * 10,
                },
                'non_loyal_threshold': {
                    'award_type_rto': lambda avg_price: round(avg_price / 10) * 10,
                    'award_type_cashback': lambda avg_price: round(avg_price / 10) * 10
                },
                'loyal_offer': {
                    'award_type_rto': lambda avg_price: round((avg_price * 0.25) * 10 / 50) * 50,
                    'award_type_cashback': lambda: 0.25
                },
                'non_loyal_offer': {
                    'award_type_rto': lambda avg_price: round((avg_price * 0.25) * 10 / 50) * 50,
                    'award_type_cashback': lambda: 0.30
                }
            },
            # Непродовольственные товары <=30% >60 руб.
            ('непродовольственные', '<=30%', '>60'): {
                'mechanics': ['award_type_rto', 'award_type_plu_count', 'award_type_cashback'],
                'loyal_threshold': {
                    'award_type_rto': lambda avg_qty: max(1, round(avg_qty * 1.2)),
                    'award_type_plu_count': lambda avg_rub: round(avg_rub * 1.2 / 10) * 10,
                    'award_type_cashback': lambda avg_rub: round(avg_rub * 1.2 / 10) * 10
                },
                'non_loyal_threshold': {
                    'award_type_rto': lambda avg_qty: max(1, round(avg_qty * 1.2)),
                    'award_type_plu_count': lambda avg_price: round(avg_price / 10) * 10,
                    'award_type_cashback': lambda: 0
                },
                'loyal_offer': {
                    'award_type_rto': lambda avg_price, plu_count: round((avg_price * 0.30 * plu_count) / 50) * 50,
                    'award_type_plu_count': lambda threshold: round((threshold * 0.30) / 50) * 50,
                    'award_type_cashback': lambda: 0.30
                },
                'non_loyal_offer': {
                    'award_type_rto': lambda avg_price: round((avg_price * 0.35) / 50) * 50,
                    'award_type_plu_count': lambda threshold: round((threshold * 0.35) / 50) * 50,
                    'award_type_cashback': lambda: 0.35
                }
            },
            # Непродовольственные товары <=30% <=60 руб.
            ('непродовольственные', '<=30%', '<=60'): {
                'mechanics': ['award_type_rto', 'award_type_cashback', 'award_type_plu_count'],
                'loyal_threshold': {
                    'award_type_rto': lambda avg_qty: max(2, round(avg_qty * 1.2)),
                    'award_type_cashback': lambda avg_rub: round(avg_rub * 1.2 / 10) * 10,
                    'award_type_plu_count': lambda avg_rub: avg_rub
                },
                'non_loyal_threshold': {
                    'award_type_rto': lambda avg_qty: max(2, round(avg_qty * 1.2)),
                    'award_type_cashback': lambda avg_price: round(avg_price / 2 / 10) * 10,
                    'award_type_plu_count': lambda avg_price: round(avg_price / 2 / 10) * 10
                },
                'loyal_offer': {
                    'award_type_rto': lambda avg_price, plu_count: round((avg_price * 0.30 * plu_count) / 50) * 50,
                    'award_type_cashback': lambda: 0.30,
                    'award_type_plu_count': lambda threshold: round((threshold * 0.30) / 50) * 50
                },
                'non_loyal_offer': {
                    'award_type_rto': lambda avg_price: round((avg_price * 0.35 * 2) / 50) * 50,
                    'award_type_cashback': lambda: 0.35,
                    'award_type_plu_count': lambda threshold: round((threshold * 0.35) / 50) * 50
                }
            },
            # Непродовольственные товары >30%
            ('непродовольственные', '>30%', 'any'): {
                'mechanics': ['award_type_plu_count', 'award_type_cashback'],
                'loyal_threshold': {
                    'award_type_plu_count': lambda avg_rub: round(avg_rub * 1.2 / 10) * 10,
                    'award_type_cashback': lambda avg_rub: round(avg_rub * 1.2 / 10) * 10
                },
                'non_loyal_threshold': {
                    'award_type_plu_count': lambda avg_rub: round(avg_rub / 10) * 10,
                    'award_type_cashback': lambda avg_rub: round(avg_rub / 10) * 10
                },
                'loyal_offer': {
                    'award_type_plu_count': lambda threshold, min_price: min(round((threshold * 0.30) / 50) * 50,
                                                                             min_price * 0.5),
                    'award_type_cashback': lambda: 0.30
                },
                'non_loyal_offer': {
                    'award_type_plu_count': lambda threshold, min_price: min(round((threshold * 0.35) / 50) * 50,
                                                                             min_price * 0.5),
                    'award_type_cashback': lambda: 0.35
                }
            }
        }

    @staticmethod
    def _get_category_key(plu_statistics):
        """Определяем ключ для правил на основе статистики"""
        avg_price = round(plu_statistics['avg_price'].mean(), 2)
        diff_price_percent = round(
            (plu_statistics['avg_price'].max() - plu_statistics['avg_price'].min()) / avg_price * 100)
        category = 'продовольственные'
        if diff_price_percent > 30:
            price_diff = '>30%'
            avg_price_range = 'any'
        else:
            price_diff = '<=30%'
            avg_price_range = '>60' if avg_price > 60 else '<=60'
        return category, price_diff, avg_price_range

    def calculate_offer(self, mechanic, is_loyal, **params):
        """Расчет порога и награды"""
        plu_statistics = pd.read_csv('plu_statistics.csv')
        params['avg_price'] = round(plu_statistics['avg_price'].mean(), 2)
        key = self._get_category_key(plu_statistics)

        if key not in self.rules:
            raise ValueError(f"Нет правила для {key}")

        rule = self.rules[key]
        loyalty_key = 'loyal' if is_loyal == 'loyal_calc' else 'non_loyal'

        # Проверяем доступность механики
        if mechanic not in rule[f'{loyalty_key}_threshold']:
            if mechanic == 'award_type_rto':
                mechanic = 'award_type_plu_count'
                print('Offer на rto невозможен, ', key)
            elif mechanic == 'award_type_plu_count':
                mechanic = 'award_type_rto'
                print('Offer на plu count невозможен, ', key)

        # Расчет порога
        threshold_func = rule[f'{loyalty_key}_threshold'][mechanic]
        sig = inspect.signature(threshold_func)
        filtered_params = {k: v for k, v in params.items() if k in sig.parameters}
        threshold = threshold_func(**filtered_params)

        # Расчет награды
        offer_func = rule[f'{loyalty_key}_offer'][mechanic]
        sig = inspect.signature(offer_func)

        # Для награды может потребоваться threshold
        offer_params = {k: v for k, v in params.items() if k in sig.parameters}
        if 'threshold' in sig.parameters:
            offer_params['threshold'] = threshold

        offer = offer_func(**offer_params)

        return {
            'threshold': threshold,
            'offer': offer,
            'mechanic': mechanic
        }

    def main(self):
        # plu_statistics = pd.read_csv('plu_statistics.csv')
        result = self.calculate_offer(
            mechanic=self.awards,
            is_loyal=self.auditorium,
            avg_qty=3.5,
            avg_rub=120
        )
        print(result)


calculater = CalculateOffer(1)
calculater.main()
