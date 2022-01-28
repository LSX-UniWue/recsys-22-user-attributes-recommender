# We all injection anntotations should inherit from this base class to ensure extendability for future use cases
class Inject:
    pass


class InjectTokenizer(Inject):
    def __init__(self, feature_name: str):
        self.feature_name = feature_name


class InjectTokenizers(Inject):
    pass


class InjectVocabularySize(Inject):
    def __init__(self, feature_name: str):
        self.feature_name = feature_name


class InjectModel(Inject):
    def __init__(self, model_cls, config_section_name: str = None):
        self.model_cls = model_cls
        self.config_section_name = config_section_name
