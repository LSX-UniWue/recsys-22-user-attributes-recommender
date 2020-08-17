from dota.purchaselog.dataset.column_dataset import ItemIdTransformer

from .mappers import EntityIdMapper

class SeqItemIdTransformer(object):
    def __init__(self, mapper: EntityIdMapper):
        self.mapper = mapper
        self.item_id_transformer = ItemIdTransformer()

    def __call__(self, raw_sesssion, **kwargs):
        purchase_log = self.item_id_transformer(raw_sesssion, **kwargs)
        return [self.mapper.get_id(item_id) for item_id in purchase_log]

