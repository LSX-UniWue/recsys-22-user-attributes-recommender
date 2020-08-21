from configs.sasrec.sas_rec_config import SASRecConfig

# FIXME: find number of heads
MOVIELENS1M_SAS_CONFIG = SASRecConfig(5, 200, 50, 2, 1, 0.2)
AMAZON_BEAUTY_SAS_CONFIG = SASRecConfig(5, 50, 50, 2, 1, 0.5)
AMAZON_GAMES_SAS_CONFIG = SASRecConfig(5, 50, 50, 2, 1, 0.5)
STEAM_SAS_CONFIG = SASRecConfig(5, 50, 50, 2, 1, 0.5)