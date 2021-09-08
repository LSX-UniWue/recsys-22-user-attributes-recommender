from asme.init.factories.modules.modules import GenericModuleFactory
from asme.losses.basket.dream.dream_loss import DreamContrastiveLoss
from asme.losses.cosrec.cosrec_loss import CosRecLoss
from asme.losses.hgn.hgn_loss import HGNLoss
from asme.losses.sasrec.sas_rec_losses import SASRecBinaryCrossEntropyLoss
from asme.models.basket.nnrec.nnrec_model import NNRecModel
from asme.models.bert4rec.bert4rec_model import BERT4RecModel
from asme.models.caser.caser_model import CaserModel
from asme.models.cosrec.cosrec_model import CosRecModel
from asme.models.hgn.hgn_model import HGNModel
from asme.models.kebert4rec.kebert4rec_model import KeBERT4RecModel
from asme.models.narm.narm_model import NarmModel
from asme.models.rnn.rnn_model import RNNModel
from asme.models.sasrec.sas_rec_model import SASRecModel
from asme.modules.baselines.bpr_module import BprModule
from asme.modules.baselines.markov_module import MarkovModule
from asme.modules.baselines.pop_module import PopModule
from asme.modules.baselines.session_pop_module import SessionPopModule
from asme.modules.masked_training_module import MaskedTrainingModule
from asme.modules.next_item_prediction_training_module import NextItemPredictionTrainingModule, \
    NextItemPredictionWithNegativeSampleTrainingModule
from asme.modules.registry import register_module, ModuleConfig
from asme.modules.sequence_next_item_prediction_training_module import SequenceNextItemPredictionTrainingModule

register_module('kebert4rec', ModuleConfig(GenericModuleFactory, MaskedTrainingModule, {
    "model_cls": KeBERT4RecModel}))

register_module('bert4rec', ModuleConfig(GenericModuleFactory, MaskedTrainingModule, {
    "model_cls": BERT4RecModel}))

register_module("caser", ModuleConfig(GenericModuleFactory, SequenceNextItemPredictionTrainingModule, {
    "model_cls": CaserModel}))

register_module("narm", ModuleConfig(GenericModuleFactory, NextItemPredictionTrainingModule, {
    "model_cls": NarmModel}))

register_module("sasrec", ModuleConfig(GenericModuleFactory, SequenceNextItemPredictionTrainingModule, {
    "model_cls": SASRecModel,
    "loss_function": SASRecBinaryCrossEntropyLoss()}))

register_module("rnn", ModuleConfig(GenericModuleFactory, NextItemPredictionTrainingModule, {
    "model_cls": RNNModel}))

register_module("cosrec", ModuleConfig(GenericModuleFactory, SequenceNextItemPredictionTrainingModule, {
    "model_cls": CosRecModel,
    "loss_function": CosRecLoss()}))

register_module("hgn", ModuleConfig(GenericModuleFactory, SequenceNextItemPredictionTrainingModule, {
    "model_cls": HGNModel,
    "loss_function": HGNLoss()}))

register_module("dream",
                ModuleConfig(GenericModuleFactory, NextItemPredictionWithNegativeSampleTrainingModule, {
                    "model_cls": RNNModel,
                    "loss_function": DreamContrastiveLoss}))

register_module("nnrec", ModuleConfig(GenericModuleFactory, NextItemPredictionTrainingModule, {
    "model_cls": NNRecModel}))

register_module("pop", ModuleConfig(GenericModuleFactory, PopModule))

register_module("session_pop", ModuleConfig(GenericModuleFactory, SessionPopModule))

register_module("markov", ModuleConfig(GenericModuleFactory, MarkovModule))

register_module("bpr", ModuleConfig(GenericModuleFactory, BprModule))
