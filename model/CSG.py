import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel, BertConfig
from transformers.models.bert.modeling_bert import BertPreTrainingHeads, BertLMPredictionHead
import torch.nn.functional as F


class CSG(nn.Module):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        **next_sentence_label**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see ``input_ids`` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates sequence B is a continuation of sequence A,
            ``1`` indicates sequence B is a random sequence.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when both ``masked_lm_labels`` and ``next_sentence_label`` are provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **seq_relationship_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, 2)``
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForPreTraining.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        prediction_scores, seq_relationship_scores = outputs[:2]

    """

    def __init__(self, plm, config, args=None):
        super(CSG, self).__init__()
        self.config = config
        self.bert = plm
        self.cls = BertPreTrainingHeads(self.config)
        self.enc = nn.Linear(768, 256)
        self.dec = nn.Linear(256, 768)
        self.enc1 = nn.Linear(768 * 2, 768)
        self.dec1 = nn.Linear(768, 768 * 2)
        # ????????????
        self.attention = nn.MultiheadAttention(config.hidden_size, 12)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.KL_loss = nn.KLDivLoss(size_average=False)
        # ?????????
        if args.pretrain_model_name == "bert-base-uncased":
            # if model_name == "roberta":
            #     config = BertConfig(len(self.word2ix))
            #     self.bert = BertModel(config)
            # dense_weight = self.bert.pooler.dense.weight
            # dense_bias = self.bert.pooler.dense.bias
            self.decoder = BertLMPredictionHead(self.config)
            # self.decoder??????BertModel??????????????????????????????????????????

    # ??????????????????loss,G_loss
    def compute_loss(self, predictions, labels, target_mask):
        """
        target_mask : ??????a?????????pad????????????0??? ?????????b?????????1
        """
        # target_mask ????????????token_type_ids, decoder?????????id???1???encoder?????????id???0?????????????????????loss
        predictions = predictions.view(-1, self.config.vocab_size)
        labels = labels.view(-1)
        target_mask = target_mask.view(-1).float()
        loss = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        return (loss(predictions, labels) * target_mask).sum() / target_mask.sum()  ## ??????mask ?????? pad ?????????a?????????????????????

    def forward(self,
                S_seq_input_ids,
                S_seq_attention_mask=None,
                S_seq_token_type_ids=None,

                S_tgt_input_ids=None,
                S_tgt_attention_mask=None,
                S_tgt_token_type_ids=None,

                C_token_type_ids=None,
                C_attention_mask=None,
                C_input_ids=None,

                G_input_tensor=None,
                G_token_type_id=None,
                G_attention_mask=None,
                G_position_enc=None,
                G_labels=None
                ):

        _, outputs_seq = self.bert(
            input_ids=S_seq_input_ids,
            attention_mask=S_seq_attention_mask,
            token_type_ids=S_seq_token_type_ids,
            # output_hidden_states=True,
            return_dict=False
        )

        _, outputs_tgt = self.bert(
            input_ids=S_tgt_input_ids,
            attention_mask=S_tgt_attention_mask,
            token_type_ids=S_tgt_token_type_ids,
            # output_hidden_states=True,
            return_dict=False
        )

        def autoencoder(input):
            input = self.enc(input)
            input = self.dec(input)
            return input

        outputs_seq = autoencoder(outputs_seq)
        KL_loss = self.KL_loss(F.log_softmax(outputs_seq / 1, dim=1), F.softmax(outputs_tgt / 1, dim=1))
        outputs_cl = self.bert(
            input_ids=C_input_ids,
            attention_mask=C_attention_mask,
            token_type_ids=C_token_type_ids,
            output_hidden_states=True,
            return_dict=True)

        # ---generation
        ## ??????????????????????????????token type id ???????????????a ?????????b????????????????????????????????????batch??????
        # Bert??????
        squence_out, _ = self.bert(
            G_input_tensor,
            position_ids=G_position_enc,
            token_type_ids=G_token_type_id,
            attention_mask=G_attention_mask,
            return_dict=False)
        ## ???????????????????????????
        # sequence_out: 3*227*768

        # ??????????????????????????????????????????????????????
        predictions = self.decoder(squence_out)
        # predictions:3, 227, 21128

        if G_labels is not None:
            ## ??????loss
            ## ???????????????????????????mask ?????????????????????loss
            # ???????????????????????????sep??????????????? ????????????-1
            predictions = predictions[:, :-1].contiguous()
            target_mask = G_token_type_id[:, 1:].contiguous()
            G_loss = self.compute_loss(predictions, G_labels, target_mask)

        return KL_loss, outputs_cl, G_loss


def GSC_loss(KL_loss, output_cl, G_loss, device, temp=0.05):
    """
    ????????????????????????
    y_pred (tensor): bert?????????, [batch_size * 2, 768]

    """
    # ??????y_pred?????????label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    y_true = torch.arange(output_cl.shape[0], device=device)
    y_true = (y_true - y_true % 2 * 2) + 1
    # batch????????????????????????, ?????????????????????(????????????)
    sim = F.cosine_similarity(output_cl.unsqueeze(1), output_cl.unsqueeze(0), dim=-1)
    # ?????????????????????????????????????????????, ?????????????????????
    sim = sim - torch.eye(output_cl.shape[0], device=device) * 1e12
    # ?????????????????????????????????
    sim = sim / temp
    # ????????????????????????y_true??????????????????
    cl_loss = F.cross_entropy(sim, y_true)
    loss = KL_loss + cl_loss + G_loss
    print("Total_loss={}. KL_loss={} ,cl_loss={} ,G_loss={}".format(loss, KL_loss, cl_loss, G_loss))
    return torch.mean(loss)
