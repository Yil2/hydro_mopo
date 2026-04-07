from pathlib import Path

import toml


class ConfigError(Exception):
    """Raised when configuration data is missing or invalid."""

class ConfigData:
    REQUIRED_CONFIG_KEYS = (
        "country_code_list",
        "hydro_type",
        "scenario",
        "algorithm",
        "data_dir",
        "solution_dir",
        "glofas_cdf_path",
    )

    def __init__(self):
        self.__PATH_USER_CONFIG = Path(__file__).parent.parent / "config_data" / "user_config.toml"
        self.__PATH_COUNTRY_MAP = Path(__file__).parent.parent / "config_data" / "country_map.toml"
        self.config = {}
        self.map = {}
        self.country_code = ""
        self.hydro_type = ""
        self.algorithm = ""
        self.__load_config()
        self.__config_check()

    def __load_config(self):
        self.config = self.__load_toml_file(self.__PATH_USER_CONFIG, "user config")
        self.map = self.__load_toml_file(self.__PATH_COUNTRY_MAP, "country map")

    def __load_toml_file(self, file_path: Path, file_label: str) -> dict:
        if not file_path.exists():
            raise ConfigError(f"Missing {file_label} file: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return toml.load(file)
        except toml.TomlDecodeError as exc:
            raise ConfigError(
                f"Invalid TOML syntax in {file_path} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
            ) from exc

    def __config_check(self):
        missing_keys = [key for key in self.REQUIRED_CONFIG_KEYS if key not in self.config]
        if missing_keys:
            missing = ", ".join(missing_keys)
            raise ConfigError(f"Missing required configuration keys: {missing}")

        for key, value in self.config.items():
            if isinstance(value, str) and value == "":
                raise ConfigError(f"TOML configuration '{key}' value is empty")

            if isinstance(value, list) and len(value) == 0:
                raise ConfigError(f"TOML configuration '{key}' value is empty")

        if not isinstance(self.config["country_code_list"], list):
            raise ConfigError("TOML configuration 'country_code_list' must be a list")

        if self.config["hydro_type"] not in {"hdam", "hror"}:
            raise ConfigError("TOML configuration 'hydro_type' must be either 'hdam' or 'hror'")

    def args_check(self, index=0):
        country_code = self.config["country_code_list"][index]
        print(f"Selected country is [{country_code}]")

        if country_code not in self.map:
            raise ConfigError(f"Input country code '{country_code}' is not available")

        hydro_type = self.config["hydro_type"]

        if hydro_type == "hdam" and self.map[country_code]["hdam_type_support"]:
            print(f"Input country code: {country_code} Select hydro type: {hydro_type}")

        elif hydro_type == "hror" and self.map[country_code]["hror_type_support"]:
            print(f"Input country code: {country_code} Select hydro type: {hydro_type}")

        else:
            raise ConfigError(
                f"Input country code [{country_code}] and hydro type [{hydro_type}] are not supported"
            )

        self.country_code = country_code
        self.hydro_type = hydro_type

class FetchPath:

    def __init__(self, config_obj):
        self.path_dict = {}
        self.__spine_gen_dir = []
        self.__set_file_path(config_obj)
        self.__create_dir()

    def __set_file_path(self, config_obj):
        country_code = config_obj.country_code
        scenario = str(config_obj.config["scenario"]).strip()
        hydro_type = config_obj.hydro_type

        if hydro_type not in {"hdam", "hror"}:
            raise ConfigError(f"Unknown hydro type: {hydro_type}")

        data_dir = Path(config_obj.config["data_dir"]).expanduser()
        history_data_path = data_dir / country_code

        self.path_dict["data_file"] = history_data_path / f"{country_code}_historical_{hydro_type}_inflow.csv"
        self.path_dict["history_data_path"] = history_data_path
        self.path_dict["glofas_cdf_path"] = config_obj.config['glofas_cdf_path']
        self.path_dict["osm_filepath"] = history_data_path / f"{country_code}_hydropower_plants.geojson"

        self.path_dict["disc_file"] = history_data_path / f"{country_code}_{hydro_type}_{scenario}_glofas_discharge.csv"

        method = str(config_obj.config["algorithm"]).strip()
        solution_dir = Path(config_obj.config["solution_dir"]).expanduser()
        self.path_dict["pred_data_path"] = solution_dir / scenario / method / "pred_data" / hydro_type
        self.path_dict["pred_data_file"] = self.path_dict["pred_data_path"] / f"{country_code}_{hydro_type}_{scenario}_modelled_data.csv"
        self.path_dict["pred_fig_path"] = solution_dir / scenario / method / "figs" / hydro_type
        self.path_dict["fitting_path"] = solution_dir / "training_cv"
        self.path_dict["fitting_result"] = self.path_dict["fitting_path"] / f"{country_code}_{hydro_type}_{method}_cv.png"
        self.__spine_gen_dir = ["history_data_path", "pred_data_path", "pred_fig_path", "fitting_path"]


    def __create_dir(self):
        for dir_key in self.__spine_gen_dir:
            self.path_dict[dir_key].mkdir(parents=True, exist_ok=True)


