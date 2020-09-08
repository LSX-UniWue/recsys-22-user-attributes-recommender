import inspect

import torch


class MultiProcessDataLoaderSupport:
    """
        Trait that is used to enable support for object recursive multi processing worker initialization in Dataset
        loading related classes.
    """
    def mp_init(self, id: int, num_worker: int, seed: int):
        """
        Recursively calls mp_init()  on all properties of type MultiProcessDataLoaderSupport. Before returning calls _mp_init().

        :param id: worker id
        :param num_worker: number of workers
        :param seed: seed
        """
        members = inspect.getmembers(self)
        for name, obj in members:
            if issubclass(type(obj), MultiProcessDataLoaderSupport):
                obj.mp_init(id, num_worker, seed)

        self._mp_init(id, num_worker, seed)

    def _mp_init(self, id: int, num_worker: int, seed: int):
        raise NotImplementedError()


def mp_worker_init_fn(id: int):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset

    # FIXME (AD) add check that dataset implements MultiProcessDataLoaderSupport
    dataset.mp_init(worker_info.id, worker_info.num_workers, worker_info.seed)
