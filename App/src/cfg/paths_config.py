import os
from pathlib import Path


__BASE_DIR__ = Path(r"./App/src").resolve()
__TRACKING_DATA_DIR__ = (__BASE_DIR__ / Path(r'tracking_data_files')).resolve()
__KAFKA_CONFIG__ = (__BASE_DIR__ / Path(r'cfg/tracking_core_kafka_config.ini')).resolve()

# print(os.path.exists(__KAFKA_CONFIG__), __KAFKA_CONFIG__.as_posix())