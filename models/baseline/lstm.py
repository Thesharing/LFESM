import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from transformers.modeling_bert import BertPreTrainedModel, BertModel

from ..esim.layers import Seq2SeqEncoder
from ..esim.utils import replace_masked


class LSTMBaseline(BertPreTrainedModel):
    """
    ab、ac交互并编码
    """

    def __init__(self, config):
        super(LSTMBaseline, self).__init__(config)
        self.bert = BertModel(config)
        self.init_weights()

        self._embedding = self.bert.embeddings.word_embeddings

        self._encoding = Seq2SeqEncoder(nn.LSTM,
                                        config.hidden_size,
                                        config.hidden_size,
                                        bidirectional=True)
        self._linear = nn.Bilinear(2 * config.hidden_size, 2 * config.hidden_size, 1)
        self.apply(self.init_esim_weights)

    def forward(self, a, b, c, labels=None, mode="prob"):
        a_mask = a[1].float()
        b_mask = b[1].float()
        c_mask = c[1].float()

        # the parameter is: input_ids, attention_mask, token_type_ids
        # which is corresponding to input_ids, input_mask and segment_ids in InputFeatures
        a_output = self._embedding(a[0])
        b_output = self._embedding(b[0])
        c_output = self._embedding(c[0])
        # The return value: sequence_output, pooled_output, (hidden_states), (attentions)

        a_length = a_mask.sum(dim=-1).long()
        b_length = b_mask.sum(dim=-1).long()
        c_length = c_mask.sum(dim=-1).long()

        v_a = self._encoding(a_output, a_length)
        v_b = self._encoding(b_output, b_length)
        v_c = self._encoding(c_output, c_length)

        v_a_max, _ = replace_masked(v_a, a_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_b, b_mask, -1e7).max(dim=1)
        v_c_max, _ = replace_masked(v_c, c_mask, -1e7).max(dim=1)

        ab = self._linear(v_a_max, v_b_max)
        ac = self._linear(v_a_max, v_c_max)

        output = torch.cat([ab, ac], dim=-1)

        if mode == "prob":
            prob = torch.nn.functional.softmax(Variable(output), dim=1)
            return prob
        elif mode == "logits":
            return output
        elif mode == "loss":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(output.view(-1, 2), labels.view(-1))
            return loss
        elif mode == "evaluate":
            prob = torch.nn.functional.softmax(Variable(output), dim=1)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(output.view(-1, 2), labels.view(-1))
            return output, prob, loss

    @staticmethod
    def init_esim_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            nn.init.constant_(module.bias.data, 0.0)
        elif isinstance(module, nn.LSTM):
            nn.init.xavier_uniform_(module.weight_ih_l0.data)
            nn.init.orthogonal_(module.weight_hh_l0.data)
            nn.init.constant_(module.bias_ih_l0.data, 0.0)
            nn.init.constant_(module.bias_hh_l0.data, 0.0)
            hidden_size = module.bias_hh_l0.data.shape[0] // 4
            module.bias_hh_l0.data[hidden_size:(2 * hidden_size)] = 1.0
            if module.bidirectional:
                nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
                nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
                nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
                nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
                module.bias_hh_l0_reverse.data[hidden_size:(2 * hidden_size)] = 1.0
