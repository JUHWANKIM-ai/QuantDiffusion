import os

LOCAL_BASE_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class quantDiffusionPath:
    def __init__(self, base_directory: str = LOCAL_BASE_DIRECTORY):
        self.base_directory = base_directory

    def get_image_path(self):
        return os.path.join(self.base_directory, f"result/images")

    def get_weight_path(self):
        return os.path.join(self.base_directory, f"result/weight")

    def get_result_path(self, sampling_type: str):
        return os.path.join(
            self.base_directory, f"result/statistics/synthetic_data/{sampling_type}"
        )

    def get_result_file(self, file_name):
        return os.path.join(self.base_directory, f"result/statistics/{file_name}.csv")

    # ---- data paths ----
    def get_data_dir(self, sub: str = ""):
        return os.path.join(self.base_directory, "data", sub)

    def get_indices_ohlcv_file(self, field: str):
        # field in {O, H, L, C, V} -> data/indices/indices_{field}.csv (raw OHLCV)
        return os.path.join(self.base_directory, "data", "indices", f"indices_{field}.csv")

    def get_jpm_file(self):
        return os.path.join(self.base_directory, "data", "JPM.csv")

    def get_indices_file(self):
        # assembled 6-asset close panel (indices_C + JPM) -> data/indices.csv
        return os.path.join(self.base_directory, "data", "indices.csv")

    def get_rv_file(self, rv_name: str):
        # data/RV/RV_{rv_name}.csv
        return os.path.join(self.base_directory, "data", "RV", f"RV_{rv_name}.csv")
