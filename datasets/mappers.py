from typing import Optional, Tuple, Dict
from pathlib import Path

import json


class EntityIdMapper:
    def __init__(self, file_path=None):
        if file_path:
            self.next_id, self.mapping = self.load(file_path)
        else:
            self.mapping = dict()
            self.next_id = 1

    def add(self, entity: int):
        if entity not in self.mapping:
            self.mapping[entity] = self.next_id
            self.next_id += 1

    def get_id(self, entity: int) -> Optional[int]:
        if entity in self.mapping:
            return self.mapping[entity]
        else:
            return None

    def get_entity(self, id: int) -> Optional[int]:
        for key, value in self.mapping.items():
            if value == id:
                return key

        return None

    def write(self, file_path: Path, overwrite: bool = False):
        if (file_path.exists() and overwrite) or not file_path.exists():
            with file_path.open(mode="w") as file:
                as_json = {
                    "next_id": self.next_id,
                    "mapping": self.mapping
                }
                json.dump(as_json, file, indent=True)
        else:
            raise Exception("File exists and overwrite is disabled.")

    @staticmethod
    def load(file_path: Path) -> Tuple[int, Dict[int, int]]:
        with file_path.open(mode="r") as file:
            loaded_json = json.load(file)
            return loaded_json["next_id"], {int(key): value for key, value in loaded_json["mapping"].items()}

    def __len__(self):
        return len(self.mapping)
