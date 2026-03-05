import os

LOCAL_BASE_DIRECTORY = (
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


class DeepHedgingPath:
    def __init__(self, base_directory: str = LOCAL_BASE_DIRECTORY):
        self.base_directory = base_directory

    def get_option_data_path(self, preprocessed: bool = False):
        if preprocessed:
            return os.path.join(self.base_directory, "data/option_data_pre")
        else:
            return os.path.join(self.base_directory, "data/option_data_ori")
    
    def get_option_data_file(self, file_name: str, preprocessed: bool = False):
        return os.path.join(self.get_option_data_path(preprocessed), file_name)
    
    def get_stock_price_data_path(self):
        return os.path.join(self.base_directory, "data/stock_price_data")

    def get_stock_price_data_file(self, file_name: str):
        return os.path.join(self.get_stock_price_data_path(), file_name)
    
    def get_option_IV_data_path(self):
        return os.path.join(self.base_directory, "data/option_IV")
    
    def get_option_IV_data_file(self, file_name: str):
        return os.path.join(self.get_option_IV_data_path(), file_name)
    
    def get_exercise_boundary_data_path(self):
        return os.path.join(self.base_directory, "data/exercise_boundary_data")

    def get_exercise_boundary_data_file(self, file_name: str):
        return os.path.join(self.get_exercise_boundary_data_path(), file_name)

    def get_synthetic_result_file(self, cp_flag: str, epi: int):
        return os.path.join(self.base_directory, f"result/statistics/EXHIBIT_5/synthetic_{cp_flag}_{epi}_result.csv")

    def get_result_path(self, table_name: str):
        return os.path.join(self.base_directory, f"result/statistics/{table_name}")

    def get_result_file(self, table_name, file_name):
        return os.path.join(self.base_directory, f"result/statistics/{table_name}/{file_name}.csv")

    def get_image_path(self, tc):
        return os.path.join(self.base_directory, f"result/images/{tc}")

    def get_weight_path(self, tc):
        return os.path.join(self.base_directory, f"result/weight/{tc}")
    
    def get_weight_temp_path(self, tc):
        return os.path.join(self.base_directory, f"result/weight_temp/{tc}")

    def get_result_excel(self, table_name, file_name):
        return os.path.join(self.base_directory, f"result/statistics/{table_name}/{file_name}.xlsx")
