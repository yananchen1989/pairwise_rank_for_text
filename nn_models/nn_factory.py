import sys
from .text_rank_dnn_model import TextRankDnnModel
from .text_rank_cnn_model import TextRankCnnModel
from .text_rank_rnn_model import TextRankRnnModel
from .text_rank_birnn_model import TextRankBiRnnModel
from .text_classify_model import TextClassifyModel
from .text_dssm_model import TextDssmModel
from .site_authority_model import SiteAuthorityModel
from .sequence_tagging_model import SequenceTaggingModel


def nn_factory(nn_name, tensor_dict, config, optimizer):
  if nn_name == "TextRankDnnModel":
    return TextRankDnnModel(tensor_dict, config, optimizer)
  elif nn_name == "TextRankCnnModel":
    return TextRankCnnModel(tensor_dict, config, optimizer)
  elif nn_name == "TextRankRnnModel":
    return TextRankRnnModel(tensor_dict, config, optimizer)
  elif nn_name == "TextRankBiRnnModel":
    return TextRankBiRnnModel(tensor_dict, config, optimizer)
  elif nn_name == "TextClassifyModel":
    return TextClassifyModel(tensor_dict, config, optimizer)
  elif nn_name == "TextDssmModel":
    return TextDssmModel(tensor_dict, config, optimizer)
  elif nn_name == "SiteAuthorityModel":
    return SiteAuthorityModel(tensor_dict, config, optimizer)
  elif nn_name == "SequenceTaggingModel":
    return SequenceTaggingModel(tensor_dict, config, optimizer)
  else:
    sys.exit('Invalid config.nn: %s' % nn_name)
