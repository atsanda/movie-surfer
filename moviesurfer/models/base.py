import os
import pickle
import typing as tp


class Parametrizable:
    def get_params(self) -> dict:
        params = {}
        attributes = vars(self)
        init_attributes = self.__init__.__code__.co_varnames[1:]
        for attribute in init_attributes:
            if attribute in attributes:
                params[attribute] = attributes[attribute]
        return params


class Serializable:
    def save(self, path: str, filename: tp.Optional[str] = None):
        filename = self.__class__.__name__ if filename is None else filename
        fp = os.path.join(path, f"{filename}.pkl")
        with open(fp, "wb") as f:
            pickle.dump(self, f)
        return fp

    def load(self, file_path: str):
        with open(file_path, "rb") as f:
            return pickle.load(f)
