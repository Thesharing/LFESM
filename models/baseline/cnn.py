import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.nn import functional as NF
from transformers.modeling_bert import BertPreTrainedModel, BertModel


class CNNBaseline(BertPreTrainedModel):
    """
    ab、ac交互并编码
    """

    def __init__(self, config):
        super(CNNBaseline, self).__init__(config)
        self.bert = BertModel(config)
        self.init_weights()

        self._embedding = self.bert.embeddings.word_embeddings

        filter_sizes = [2, 3, 4, 5]
        num_filters = 36
        embedding_size = 768

        self._convs = nn.ModuleList([nn.Conv2d(1, num_filters, (K, embedding_size)) for K in filter_sizes])
        self._dropout = nn.Dropout(0.1)
        self._linear = nn.Bilinear(len(filter_sizes) * num_filters, len(filter_sizes) * num_filters, 1)
        self.apply(self.init_esim_weights)

    def _encoding(self, a):
        x = self._embedding(a[0])
        x = x.unsqueeze(1)
        x = [NF.relu(conv(x).squeeze(3)) for conv in self._convs]
        x = [NF.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self._dropout(x)
        return x

    def forward(self, a, b, c, labels=None, mode="prob"):
        v_a = self._encoding(a)
        v_b = self._encoding(b)
        v_c = self._encoding(c)

        ab = self._linear(v_a, v_b)
        ac = self._linear(v_a, v_c)

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
