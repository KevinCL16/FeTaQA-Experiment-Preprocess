import torch
import pickle
import torch.nn as nn


class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.view(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.view(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        output = torch.mul(attention_weights.transpose(1, 2), context)

        return output, attention_weights


# open
with open('./representation/row_representation.pickle', 'rb') as handle:
    row_representation = pickle.load(handle)
with open('./representation/col_representation.pickle', 'rb') as handle:
    col_representation = pickle.load(handle)
with open('./representation/grid_representation.pickle', 'rb') as handle:
    grid_representation = pickle.load(handle)
with open('./representation/tcq_representation.pickle', 'rb') as handle:
    table_context_question_representation = pickle.load(handle)

attention_calc = Attention(768, attention_type='dot')
attention_calc.to('cuda:0')

weighted_row_representation = []
weighted_col_representation = []
weighted_grid_representation = []

for idx in range(len(row_representation)):
    weighted_row_rep, _ = attention_calc(table_context_question_representation[idx].unsqueeze(dim=0),
                                         row_representation[idx].to('cuda:0').unsqueeze(dim=0))
    weighted_row_representation.append(weighted_row_rep)

for idx in range(len(col_representation)):
    weighted_col_rep, _ = attention_calc(table_context_question_representation[idx].unsqueeze(dim=0),
                                         col_representation[idx].unsqueeze(dim=0))
    weighted_col_representation.append(weighted_col_rep)

for idx in range(len(grid_representation)):
    for j in range(len(grid_representation[idx]) - 1):
        grid_tensor_in_grid = grid_representation[idx][j][0]
        for i in range(len(grid_representation[idx][0]) - 1):
            temp = grid_representation[idx][j][i + 1].clone().detach()
            grid_tensor_in_grid = torch.cat((temp, grid_tensor_in_grid), dim=-2)
        if j == 0:
            grid_tensor_pre_attention = grid_tensor_in_grid.unsqueeze(dim=0)
        grid_tensor_pre_attention = torch.cat((grid_tensor_in_grid.unsqueeze(dim=0), grid_tensor_pre_attention), dim=0)
    grid_tensor_pre_attention = torch.reshape(grid_tensor_pre_attention,
                                              [len(grid_representation[idx]) * len(grid_representation[idx][0]), 768])

    weighted_grid_rep, _ = attention_calc(table_context_question_representation[idx].unsqueeze(dim=0),
                                          grid_tensor_pre_attention.unsqueeze(dim=0))

    weighted_grid_representation.append(weighted_grid_rep)

with open('./weighted_representation/weighted_col_representation.pickle', 'wb') as handle:
    pickle.dump(weighted_col_representation, handle)

with open('./weighted_representation/weighted_row_representation.pickle', 'wb') as handle:
    pickle.dump(weighted_row_representation, handle)

with open('./weighted_representation/weighted_grid_representation.pickle', 'wb') as handle:
    pickle.dump(weighted_grid_representation, handle)
