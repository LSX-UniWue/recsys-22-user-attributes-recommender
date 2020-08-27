
from pathlib import Path

from data.base.reader import CsvSessionDatasetReader, Index
from data.datasets.nextitem import NextItemPredSessionIndex, NextItemPredSessionDataset
from data.datasets.seqitem import SequentialItemSessionDataset, SequentialItemSessionParser
from data.utils import create_indexed_header, read_csv_header


def main_sequential_item_session_dataset():
    base = Path("/home/dallmann/uni/research/dota/datasets/small")
    data_file_path = base / "small.csv"
    index_file_path = base / "small.idx"

    delimiter = "\t"

    reader = CsvSessionDatasetReader(data_file_path, Index(index_file_path))

    dataset = SequentialItemSessionDataset(
        reader,
        SequentialItemSessionParser(
            create_indexed_header(
                read_csv_header(data_file_path, delimiter=delimiter)
            ),
            "item_id", delimiter=delimiter
        )
    )

    print(dataset[0])


def main_next_item_prediction_session_dataset():
    base = Path("/home/dallmann/uni/research/dota/datasets/small")
    data_file_path = base / "small.csv"
    index_file_path = base / "small.idx"
    next_item_index_file_path = base / "small_nip.idx"
    delimiter = "\t"

    reader = CsvSessionDatasetReader(data_file_path, Index(index_file_path))

    dataset = SequentialItemSessionDataset(
        reader,
        SequentialItemSessionParser(
            create_indexed_header(
                read_csv_header(data_file_path, delimiter=delimiter)
            ),
            "item_id", delimiter=delimiter
        )
    )

    seq_item_index = NextItemPredSessionIndex(next_item_index_file_path)
    seq_item_dataset = NextItemPredSessionDataset(dataset, seq_item_index)

    print(seq_item_dataset[10])


if __name__ == "__main__":
    main_sequential_item_session_dataset()
    #main_next_item_prediction_session_dataset()

# def next_item_pred_iterable_dataset_initfn(worker_id):
#     worker_info = torch.utils.data.get_worker_info()
#     dataset = worker_info.dataset
#     num_workers = worker_info.num_workers
#     # make sure that all file handles etc. are correctly initialized
#     dataset.reinit()
#
#     worker_share = int(math.ceil(len(dataset._session_dataset) / float(num_workers)))
#
#     start = worker_id * worker_share
#     if worker_id < num_workers - 1:
#         end = min(start + worker_share - 1, len(dataset._session_dataset))
#     else:
#         end = len(dataset._session_dataset)
#
#     dataset.start = start
#     dataset.end = end
#     dataset.seed = worker_info.seed
#
#
# class NextItemPredIterableDataset(IterableDataset):
#
#     def __init__(self, session_dataset: SessionDataset, start: int, end: int, seed: int):
#         self._session_dataset = session_dataset
#         self.start = start
#         self.end = end
#         self.seed = seed
#
#     def reinit(self):
#         self._session_dataset.reinit()
#
#     def __iter__(self):
#         num_samples = self.end - self.start
#         rng = default_rng(self.seed)
#         while True:
#             session_idx = self.start + rng.integers(low=0, high=num_samples)
#             session = self._session_dataset[session_idx]
#
#             if len(session) < 2:
#                 continue
#
#             target_pos = rng.integers(low=1, high=len(session))
#
#             yield {
#                 "session": session[:target_pos],
#                 "target": session[target_pos]
#             }


# class NegSamplingPurchaseLogDataset(Dataset):
#
#     def __init__(self, pl_dataset: SessionDataset, seed: int = 123456):
#         self._pl_dataset = pl_dataset
#         self._num_users = len(pl_dataset)
#         self._num_items = len(pl_dataset.item_id_mapper)
#
#         self._rng = default_rng(seed)
#
#     def __len__(self):
#         return len(self._pl_dataset)
#
#     def __getitem__(self, idx):
#         purchase_log = self._pl_dataset.__getitem__(idx)
#         # we can't predict first item, since we need to be able to build a "user" embedding
#         # in validation and test, so we will skip it in general.
#         pos_item_idx = self._rng.integers(low=1, high=len(purchase_log))
#         pos_item = purchase_log[pos_item_idx]
#
#         session = purchase_log[:pos_item_idx]
#
#         # sample candidates from all items that are not part of the session
#         candidates = [x for x in range(self._num_items) if x not in purchase_log]
#         neg_item_idx = self._rng.integers(len(candidates))
#         neg_item = candidates[neg_item_idx]
#
#         # return the "user" substituted by the session id, since we have anonymous users
#         # the session: all items before the positive sample
#         # a randomly sampled negative item that does not occur in the session
#         return {
#             "user": idx,
#             "session": session,
#             "pos": pos_item,
#             "neg": neg_item
#         }
#
#
# def test_next_item_iterable():
#     seed = 98393939
#     base = Path("/home/dallmann/uni/research/dota/datasets/small")
#     ds = SessionDataset(base / "small.csv", base / "small.idx")
#     data = NextItemPredIterableDataset(ds, 0, len(ds), seed)
#
#     loader = DataLoader(data, batch_size=512, shuffle=False, worker_init_fn=next_item_pred_iterable_dataset_initfn, num_workers=4)
#
#     for sample in tqdm(loader, desc="Benchmarking Dataset"):
#     #for sample in loader:
#         pass
#
#
# def testFullSamplingIndex():
#     seed = 98393939
#     base = Path("/home/dallmann/uni/research/dota/datasets/small")
#     ds = SessionDataset(base / "small.csv", base / "small.idx")
#     index = NextItemPredSessionIndex(ds, base / "small_nip.idx")
#     data = NextItemPredSessionDataset(ds, index)
#
#     loader = DataLoader(data, batch_size=512, shuffle=False, num_workers=1)
#     counter = 0
#
#     for sample in tqdm(loader, desc="Benchmarking Dataset"):
#        pass
#         #counter += 1
#         #if counter == 10:
#         #    break
#         #pass
#
#
# def test():
#     seed = 98393939
#     base = Path("/home/dallmann/uni/research/dota/datasets/small")
#     ds = SessionDataset(base / "small.csv", base / "small.idx")
#     neg_sampling_ds = NegSamplingPurchaseLogDataset(ds, seed=seed)
#
#     counter = 0
#     for sample in neg_sampling_ds:
#         print(sample)
#         counter += 1
#         if counter == 10:
#             break
#
#
# def max_length():
#     base = Path("/home/dallmann/uni/research/dota/datasets/small")
#     ds = SessionDataset(base / "small.csv", base / "small.idx")
#
#     max = 0
#     for idx in tqdm(range(len(ds))):
#         log = ds.__getitem__(idx)
#         if len(log) > max:
#             max = len(log)
#
#     print(max)
#
#
# if __name__ == "__main__":
#     #test()
#     #max_length()
#     #testFullSamplingIndex()
#     test_next_item_iterable()