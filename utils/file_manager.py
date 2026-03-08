import os

LOCAL_BASE_DIRECTORY = (
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

class quantDiffusionPath:
    def __init__(self, base_directory: str = LOCAL_BASE_DIRECTORY):
        self.base_directory = base_directory

    def get_image_path(self):
        return os.path.join(self.base_directory, f"result/images")

    def get_weight_path(self):
        return os.path.join(self.base_directory, f"result/weight")

    def get_result_path(self, sampling_type: str):
        return os.path.join(self.base_directory, f"result/statistics/synthetic_data/{sampling_type}")

    def get_result_file(self, file_name):
        return os.path.join(self.base_directory, f"result/statistics/{file_name}.csv")
