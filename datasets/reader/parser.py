import csv
import io
from typing import List, Any, Dict


class SessionParser(object):
    def parse(self, columns: Dict[str, int], raw_session: io.StringIO) -> Dict[str, List[Any]]:
        pass


class ItemIdSessionParser(SessionParser):

    def __init__(self, item_id_column_name: str):
        self.item_id_column_name = item_id_column_name

    def parse(self, columns: Dict[str, int], raw_sesssion: io.StringIO) -> Dict[str, List[Any]]:

        reader = csv.reader(raw_sesssion, delimiter="\t")
        session = list()
        for line in reader:
            item_id = int(line[columns[self.item_id_column_name]])
            session.append(item_id)

        return {"session": session}
