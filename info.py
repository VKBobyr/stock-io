import os


class Paths:
    folder_data = 'data/'
    extracted_stocks = os.path.join(folder_data, 'stocks_extracted')
    sector_stock_dictionary = os.path.join(folder_data, 'tickers_by_sector.json')
    ticker_num_entries = os.path.join(folder_data, 'ticker_num_entries.json')


class DataKeys:
    sector_comm = 'sector_communication'
    sector_cons_disc = 'sector_consumer_discretionary'
    sector_energy = 'sector_energy'
    sector_financial = 'sector_financial'
    sector_health = 'sector_healthcare'
    sector_industrial = 'sector_industrial'
    sector_materials = 'sector_materials'
    sector_real_estate = 'sector_real_estate'
    sector_tech = 'sector_technology'
    sector_utility = 'sector_utility'

    type_stocks = 0
    type_etfs = 1
    type_etfs_and_stocks = 2

    price_open = 'price_open'
    price_close = 'price_close'
    price_high = 'price_high'
    price_low = 'price_low'

    stat_volume = 'stat_volume'

    dt_month_of_year = 'month_of_year'
    dt_day_of_month = 'day_of_month'
    dt_day_of_week = 'day_of_week'

    use = 'use'

    data_size = 'size'


class Constants:
    epsilon = 1e-6
    days_in_week = 5
    days_in_month = 31
    months_in_year = 12


class DataType:
    def __init__(self, used: bool, dimension: int, is_categorical=False):
        self.used = used
        self.dimension = dimension
        self.is_categorical = is_categorical


class ModelParameters:
    model_name = 'baconator'

    days_to_predict = sorted([1, 2])
    output_days = len(days_to_predict)
    input_days = 256
    validation_interval = input_days + max(days_to_predict) + 100
    min_data_size = 1200

    predicted_key = DataKeys.price_close
    predict_direction = True

    if predict_direction:
        losses = ["binary_crossentropy"] * len(days_to_predict)
    else:
        losses = ["mean_squared_error"] * len(days_to_predict)
    metrics = ["accuracy"] * len(days_to_predict)

    batch_size = 256
    learn_rate = 1e-4

    # data ripcords
    dk = DataKeys
    data = {
        dk.price_open: DataType(used=True, dimension=1),
        dk.price_close: DataType(used=True, dimension=1),
        dk.price_low: DataType(used=True, dimension=1),
        dk.price_high: DataType(used=True, dimension=1),
        dk.stat_volume: DataType(used=True, dimension=1),
        dk.dt_day_of_week: DataType(used=True, dimension=Constants.days_in_week, is_categorical=True),
        dk.dt_day_of_month: DataType(used=True, dimension=Constants.days_in_month, is_categorical=True),
        dk.dt_month_of_year: DataType(used=True, dimension=Constants.months_in_year, is_categorical=True),
    }

    info_dimension = 0
    for data_type in data.values():
        if not data_type.used:
            continue

        info_dimension += data_type.dimension
