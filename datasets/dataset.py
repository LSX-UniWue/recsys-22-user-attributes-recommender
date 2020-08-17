import io
import math
from pathlib import Path
import sys

import torch
from numpy.random import default_rng
from torch.utils.data import Dataset, IterableDataset
from dota.purchaselog.dataset.column_dataset import Index, ColumnDataset
from torch.utils.data.dataloader import DataLoader
import torch
from tqdm import tqdm

from datasets.transformers import SeqItemIdTransformer
from datasets.mappers import EntityIdMapper

INT_BYTE_SIZE = 8


class SessionDataset(Dataset):
    def __init__(self, data_file_path: Path, index_file_path: Path, item_seqid_mapping_path: Path = None):
        self._data_file_path = data_file_path
        self._index_file_path = index_file_path
        self._item_seqid_mapping_path = item_seqid_mapping_path

        self.reinit()

    def reinit(self):
        index = Index(self._index_file_path)

        if self._item_seqid_mapping_path:
            self._item_id_mapper = EntityIdMapper(self._item_seqid_mapping_path)
        else:
            item_id_mapper_path = self._data_file_path.parent / f"{self._data_file_path.name}.items.idx"

            # maybe it has already been created, then reuse:
            if item_id_mapper_path.exists():
                self._item_id_mapper = EntityIdMapper(item_id_mapper_path)
            else:
                self._item_id_mapper = self.build_item_id_mapper(self._data_file_path, self._index_file_path)
                self._item_id_mapper.write(item_id_mapper_path)

        self.ds = ColumnDataset(self._data_file_path, index, SeqItemIdTransformer(self._item_id_mapper))

    @staticmethod
    def build_item_id_mapper(data_file_path: Path, index_file_path: Path) -> EntityIdMapper:
        dataset = ColumnDataset(data_file_path, Index(index_file_path))
        mapper = EntityIdMapper()

        for idx in tqdm(range(len(dataset))):
            for item_id in dataset.get_session(idx):
                mapper.add(item_id)

        return mapper

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds.get_session(idx)

    @property
    def item_id_mapper(self):
        return self._item_id_mapper


class PartialSessionDataset(Dataset):
    def __init__(self, ds: SessionDataset, start: int, stop: int):
        self._ds = ds
        self._start = start
        self._stop = stop
        self._length = stop - start

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        return self._ds.__getitem__(self._start + idx)

    @property
    def item_id_mapper(self):
        return self._ds.item_id_mapper


class NextItemPredSessionIndex:
    def __init__(self, dataset: SessionDataset, index_path: Path, min_session_length: int = 2, save_index: bool = True):

        if not index_path.exists() and save_index:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            self._create(dataset, index_path, min_session_length)

        self._index_file_handle = index_path.open("rb")

        self._min_session_length = self._read_min_session_length()
        self._length = self._read_length()

    def _create(self, dataset: SessionDataset, index_path: Path, min_session_length: int):
        current_idx = 0
        with index_path.open("wb") as index_file:
            for session_idx in tqdm(range(len(dataset)), desc="Creating Index."):
                purchase_log = dataset[session_idx]
                if len(purchase_log) > min_session_length:
                    for target_pos in range(1, len(purchase_log)):
                        self._write_entry(index_file, session_idx, target_pos)
                        current_idx += 1
            # write length at the end
            index_file.write(current_idx.to_bytes(INT_BYTE_SIZE, byteorder=sys.byteorder, signed=False))
            # write minimum length
            index_file.write(min_session_length.to_bytes(INT_BYTE_SIZE, byteorder=sys.byteorder, signed=False))

    @staticmethod
    def _write_entry(index_file, session_idx: int, target_pos: int):
        index_file.write(session_idx.to_bytes(INT_BYTE_SIZE, byteorder=sys.byteorder, signed=False))
        index_file.write(target_pos.to_bytes(INT_BYTE_SIZE, byteorder=sys.byteorder, signed=False))

    def _read_length(self):
        self._index_file_handle.seek(-2*INT_BYTE_SIZE, io.SEEK_END)
        return int.from_bytes(self._index_file_handle.read(INT_BYTE_SIZE), byteorder=sys.byteorder, signed=False)

    def _read_min_session_length(self):
        self._index_file_handle.seek(-INT_BYTE_SIZE, io.SEEK_END)
        return int.from_bytes(self._index_file_handle.read(INT_BYTE_SIZE), byteorder=sys.byteorder, signed=False)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        self._index_file_handle.seek(idx * INT_BYTE_SIZE * 2)
        session_idx = int.from_bytes(self._index_file_handle.read(INT_BYTE_SIZE), byteorder=sys.byteorder, signed=False)
        target_pos = int.from_bytes(self._index_file_handle.read(INT_BYTE_SIZE), byteorder=sys.byteorder, signed=False)

        return session_idx, target_pos


class NextItemPredSessionDataset(Dataset):
    def __init__(self, dataset: SessionDataset, index: NextItemPredSessionIndex):
        super(NextItemPredSessionDataset, self).__init__()
        self.dataset = dataset
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        session_idx, target_pos = self.index[idx]
        session = self.dataset[session_idx]

        return {
            "session": session[:target_pos],
            "target": session[target_pos]
        }


def next_item_pred_iterable_dataset_initfn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    num_workers = worker_info.num_workers
    # make sure that all file handles etc. are correctly initialized
    dataset.reinit()

    worker_share = int(math.ceil(len(dataset._session_dataset) / float(num_workers)))

    start = worker_id * worker_share
    if worker_id < num_workers - 1:
        end = min(start + worker_share - 1, len(dataset._session_dataset))
    else:
        end = len(dataset._session_dataset)

    dataset.start = start
    dataset.end = end
    dataset.seed = worker_info.seed


class NextItemPredIterableDataset(IterableDataset):

    def __init__(self, session_dataset: SessionDataset, start: int, end: int, seed: int):
        self._session_dataset = session_dataset
        self.start = start
        self.end = end
        self.seed = seed

    def reinit(self):
        self._session_dataset.reinit()

    def __iter__(self):
        num_samples = self.end - self.start
        rng = default_rng(self.seed)
        while True:
            session_idx = self.start + rng.integers(low=0, high=num_samples)
            session = self._session_dataset[session_idx]

            if len(session) < 2:
                continue

            target_pos = rng.integers(low=1, high=len(session))

            yield {
                "session": session[:target_pos],
                "target": session[target_pos]
            }


class NegSamplingPurchaseLogDataset(Dataset):

    def __init__(self, pl_dataset: SessionDataset, seed: int = 123456):
        self._pl_dataset = pl_dataset
        self._num_users = len(pl_dataset)
        self._num_items = len(pl_dataset.item_id_mapper)

        self._rng = default_rng(seed)

    def __len__(self):
        return len(self._pl_dataset)

    def __getitem__(self, idx):
        purchase_log = self._pl_dataset.__getitem__(idx)
        # we can't predict first item, since we need to be able to build a "user" embedding
        # in validation and test, so we will skip it in general.
        pos_item_idx = self._rng.integers(low=1, high=len(purchase_log))
        pos_item = purchase_log[pos_item_idx]

        session = purchase_log[:pos_item_idx]

        # sample candidates from all items that are not part of the session
        candidates = [x for x in range(self._num_items) if x not in purchase_log]
        neg_item_idx = self._rng.integers(len(candidates))
        neg_item = candidates[neg_item_idx]

        # return the "user" substituted by the session id, since we have anonymous users
        # the session: all items before the positive sample
        # a randomly sampled negative item that does not occur in the session
        return {
            "user": idx,
            "session": session,
            "pos": pos_item,
            "neg": neg_item
        }


def test_next_item_iterable():
    seed = 98393939
    base = Path("/home/dallmann/uni/research/dota/datasets/small")
    ds = SessionDataset(base / "small.csv", base / "small.idx")
    data = NextItemPredIterableDataset(ds, 0, len(ds), seed)

    loader = DataLoader(data, batch_size=512, shuffle=False, worker_init_fn=next_item_pred_iterable_dataset_initfn, num_workers=4)

    for sample in tqdm(loader, desc="Benchmarking Dataset"):
    #for sample in loader:
        pass


def testFullSamplingIndex():
    seed = 98393939
    base = Path("/home/dallmann/uni/research/dota/datasets/small")
    ds = SessionDataset(base / "small.csv", base / "small.idx")
    index = NextItemPredSessionIndex(ds, base / "small_nip.idx")
    data = NextItemPredSessionDataset(ds, index)

    loader = DataLoader(data, batch_size=512, shuffle=False, num_workers=1)
    counter = 0

    for sample in tqdm(loader, desc="Benchmarking Dataset"):
       pass
        #counter += 1
        #if counter == 10:
        #    break
        #pass


def test():
    seed = 98393939
    base = Path("/home/dallmann/uni/research/dota/datasets/small")
    ds = SessionDataset(base / "small.csv", base / "small.idx")
    neg_sampling_ds = NegSamplingPurchaseLogDataset(ds, seed=seed)

    counter = 0
    for sample in neg_sampling_ds:
        print(sample)
        counter += 1
        if counter == 10:
            break


def max_length():
    base = Path("/home/dallmann/uni/research/dota/datasets/small")
    ds = SessionDataset(base / "small.csv", base / "small.idx")

    max = 0
    for idx in tqdm(range(len(ds))):
        log = ds.__getitem__(idx)
        if len(log) > max:
            max = len(log)

    print(max)


if __name__ == "__main__":
    #test()
    #max_length()
    #testFullSamplingIndex()
    test_next_item_iterable()