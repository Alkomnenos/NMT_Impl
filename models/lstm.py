import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq import utils
from seq2seq.models import Seq2SeqModel, Seq2SeqEncoder, Seq2SeqDecoder
from seq2seq.models import register_model, register_model_architecture


@register_model('lstm')
class LSTMModel(Seq2SeqModel):
    def __init__(self,
                 encoder,
                 decoder):

        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser."""
        parser.add_argument('--encoder-embed-dim', type=int)
        parser.add_argument('--encoder-embed-path')
        parser.add_argument('--encoder-hidden-size', type=int)
        parser.add_argument('--encoder-num-layers', type=int)
        parser.add_argument('--encoder-bidirectional')
        parser.add_argument('--encoder-dropout-in')
        parser.add_argument('--encoder-dropout-out')

        parser.add_argument('--decoder-embed-dim', type=int)
        parser.add_argument('--decoder-embed-path')
        parser.add_argument('--decoder-hidden-size', type=int)
        parser.add_argument('--decoder-num-layers', type=int)
        parser.add_argument('--decoder-dropout-in', type=float)
        parser.add_argument('--decoder-dropout-out', type=float)
        parser.add_argument('--decoder-use-attention')
        parser.add_argument('--decoder-use-lexical-model')

    @classmethod
    def build_model(cls, args, src_dict, tgt_dict):
        """ Constructs the model. """
        base_architecture(args)
        encoder_pretrained_embedding = None
        decoder_pretrained_embedding = None

        # Load pre-trained embeddings
        if args.encoder_embed_path:
            encoder_pretrained_embedding = utils.load_embedding(args.encoder_embed_path, src_dict)
        if args.decoder_embed_path:
            decoder_pretrained_embedding = utils.load_embedding(args.decoder_embed_path, tgt_dict)

        # encoder construction
        encoder = LSTMEncoder(dictionary=src_dict,
                              embed_dim=args.encoder_embed_dim,
                              hidden_size=args.encoder_hidden_size,
                              num_layers=args.encoder_num_layers,
                              bidirectional=bool(eval(args.encoder_bidirectional)),
                              dropout_in=args.encoder_dropout_in,
                              dropout_out=args.encoder_dropout_out,
                              pretrained_embedding=encoder_pretrained_embedding)

        # decoder construction
        decoder = LSTMDecoder(dictionary=tgt_dict,
                              embed_dim=args.decoder_embed_dim,
                              hidden_size=args.decoder_hidden_size,
                              num_layers=args.decoder_num_layers,
                              dropout_in=args.decoder_dropout_in,
                              dropout_out=args.decoder_dropout_out,
                              pretrained_embedding=decoder_pretrained_embedding,
                              use_attention=bool(eval(args.decoder_use_attention)),
                              use_lexical_model=bool(eval(args.decoder_use_lexical_model)))
        return cls(encoder, decoder)


class LSTMEncoder(Seq2SeqEncoder):
    """ Defines the encoder class. """

    def __init__(self,
                 dictionary,
                 embed_dim=64,
                 hidden_size=64,
                 num_layers=1,
                 bidirectional=True,
                 dropout_in=0.25,
                 dropout_out=0.25,
                 pretrained_embedding=None):

        super().__init__(dictionary)
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.output_dim = 2 * hidden_size if bidirectional else hidden_size

        if pretrained_embedding is not None:
            self.embedding = pretrained_embedding
        else:
            self.embedding = nn.Embedding(len(dictionary), embed_dim, dictionary.pad_idx)

        dropout_lstm = dropout_out if num_layers > 1 else 0.
        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout_lstm,
                            bidirectional=bool(bidirectional))

    def forward(self, src_tokens, src_lengths):

        # Embed tokens and apply dropout
        batch_size, src_time_steps = src_tokens.size()
        src_embeddings = self.embedding(src_tokens)
        _src_embeddings = F.dropout(src_embeddings, p=self.dropout_in, training=self.training)

        # Transpose batch: [batch_size, src_time_steps, num_features] -> [src_time_steps, batch_size, num_features]
        src_embeddings = _src_embeddings.transpose(0, 1)

        # Convert embedded tokens into a PackedSequence
        packed_source_embeddings = nn.utils.rnn.pack_padded_sequence(src_embeddings, src_lengths.data.tolist())

        # Pass source input through the recurrent layer
        if self.bidirectional:
            state_size = 2 * self.num_layers, batch_size, self.hidden_size
        else:
            state_size = self.num_layers, batch_size, self.hidden_size

        hidden_initial = src_embeddings.new_zeros(*state_size)
        context_initial = src_embeddings.new_zeros(*state_size)

        packed_outputs, (final_hidden_states, final_cell_states) = self.lstm(packed_source_embeddings,
                                                                             (hidden_initial, context_initial))

        # Unpack outputs and apply dropout
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, padding_value=0.)
        lstm_output = F.dropout(lstm_output, p=self.dropout_out, training=self.training)
        assert list(lstm_output.size()) == [src_time_steps, batch_size, self.output_dim]  # sanity check

        if self.bidirectional:
            def combine_directions(outs):
                return torch.cat([outs[0: outs.size(0): 2], outs[1: outs.size(0): 2]], dim=2)

            final_hidden_states = combine_directions(final_hidden_states)
            final_cell_states = combine_directions(final_cell_states)

        # Generate mask zeroing-out padded positions in encoder inputs
        src_mask = src_tokens.eq(self.dictionary.pad_idx)

        return {'src_embeddings': _src_embeddings.transpose(0, 1),
                'src_out': (lstm_output, final_hidden_states, final_cell_states),
                'src_mask': src_mask if src_mask.any() else None}


class AttentionLayer(nn.Module):
    """ Luong's global attention, general scoring function """

    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.src_projection = nn.Linear(input_dims, output_dims, bias=False)
        self.context_plus_hidden_projection = nn.Linear(input_dims + output_dims, output_dims, bias=False)

    def forward(self, tgt_input, encoder_out, src_mask):
        encoder_out = encoder_out.transpose(1, 0)
        attn_scores = self.score(tgt_input, encoder_out)

        if src_mask is not None:
            src_mask = src_mask.unsqueeze(dim=1)
            # src_mask.size = [batch_size,1,src_time_steps]
            attn_scores.masked_fill_(src_mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_context = torch.bmm(attn_weights, encoder_out).squeeze(dim=1)
        context_plus_hidden = torch.cat([tgt_input, attn_context], dim=1)
        attn_out = torch.tanh(self.context_plus_hidden_projection(context_plus_hidden))

        return attn_out, attn_weights.squeeze(dim=1)

    def score(self, tgt_input, encoder_out):
        projected_encoder_out = self.src_projection(encoder_out).transpose(2, 1)
        attn_scores = torch.bmm(tgt_input.unsqueeze(dim=1), projected_encoder_out)

        return attn_scores


class LSTMDecoder(Seq2SeqDecoder):

    def __init__(self,
                 dictionary,
                 embed_dim=64,
                 hidden_size=128,
                 num_layers=1,
                 dropout_in=0.25,
                 dropout_out=0.25,
                 pretrained_embedding=None,
                 use_attention=True,
                 use_lexical_model=False):

        super().__init__(dictionary)

        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size

        if pretrained_embedding is not None:
            self.embedding = pretrained_embedding
        else:
            self.embedding = nn.Embedding(len(dictionary), embed_dim, dictionary.pad_idx)

        # Define decoder layers and modules
        self.attention = AttentionLayer(hidden_size, hidden_size) if use_attention else None

        self.layers = nn.ModuleList([nn.LSTMCell(
            input_size=hidden_size + embed_dim if layer == 0 else hidden_size,
            hidden_size=hidden_size)
            for layer in range(num_layers)])

        self.final_projection = nn.Linear(hidden_size, len(dictionary))

        self.use_lexical_model = use_lexical_model

        if self.use_lexical_model:
            self.lexical_embedding_projection = nn.Linear(embed_dim, hidden_size)
            self.lexical_output_projection = nn.Linear(hidden_size, len(dictionary))

    def forward(self, tgt_inputs, encoder_out, incremental_state=None):

        # feed decoder input token-by-token if incremental generation
        if incremental_state is not None:
            tgt_inputs = tgt_inputs[:, -1:]

        src_embeddings = encoder_out['src_embeddings']

        src_out, src_hidden_states, src_cell_states = encoder_out['src_out']
        src_mask = encoder_out['src_mask']
        src_time_steps = src_out.size(0)

        # Embed target tokens and apply dropout
        batch_size, tgt_time_steps = tgt_inputs.size()
        tgt_embeddings = self.embedding(tgt_inputs)
        tgt_embeddings = F.dropout(tgt_embeddings, p=self.dropout_in, training=self.training)

        # Transpose batch: [batch_size, tgt_time_steps, num_features] -> [tgt_time_steps, batch_size, num_features]
        tgt_embeddings = tgt_embeddings.transpose(0, 1)

        # Initialize previous states (or retrieve from cache during incremental generation)

        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            tgt_hidden_states, tgt_cell_states, input_feed = cached_state
        else:
            tgt_hidden_states = [torch.zeros(tgt_inputs.size()[0], self.hidden_size) for i in range(len(self.layers))]
            tgt_cell_states = [torch.zeros(tgt_inputs.size()[0], self.hidden_size) for i in range(len(self.layers))]
            input_feed = tgt_embeddings.data.new(batch_size, self.hidden_size).zero_()

        # Initialize attention output node
        attn_weights = tgt_embeddings.data.new(batch_size, tgt_time_steps, src_time_steps).zero_()
        rnn_outputs = []

        lexical_contexts = []

        for j in range(tgt_time_steps):
            # input feeding
            lstm_input = torch.cat([tgt_embeddings[j, :, :], input_feed], dim=1)

            for layer_id, rnn_layer in enumerate(self.layers):
                # Pass target input through recurrent layer
                tgt_hidden_states[layer_id], tgt_cell_states[layer_id] = \
                    rnn_layer(lstm_input, (tgt_hidden_states[layer_id], tgt_cell_states[layer_id]))

                lstm_input = F.dropout(tgt_hidden_states[layer_id], p=self.dropout_out, training=self.training)

            if self.attention is None:
                input_feed = tgt_hidden_states[-1]
            else:
                input_feed, step_attn_weights = self.attention(tgt_hidden_states[-1], src_out, src_mask)
                attn_weights[:, j, :] = step_attn_weights

                if self.use_lexical_model:
                    src_embeddings_transposed = src_embeddings.transpose(0, 1)
                    attn_weights_expanded = step_attn_weights.unsqueeze(dim=-1)

                    weighted_source_embeddings = attn_weights_expanded * src_embeddings_transposed
                    context_vector_summed = torch.sum(weighted_source_embeddings, dim=1)

                    lexical_context_vector = torch.tanh(context_vector_summed)
                    lexical_contexts.append(lexical_context_vector)

            input_feed = F.dropout(input_feed, p=self.dropout_out, training=self.training)
            rnn_outputs.append(input_feed)

        # only used during auto-regressive generation
        utils.set_incremental_state(
            self, incremental_state, 'cached_state', (tgt_hidden_states, tgt_cell_states, input_feed))

        # Collect outputs
        decoder_output = torch.cat(rnn_outputs, dim=0).view(tgt_time_steps, batch_size, self.hidden_size)

        # Transpose batch back: [tgt_time_steps, batch_size, num_features] -> [batch_size, tgt_time_steps, num_features]
        decoder_output = decoder_output.transpose(0, 1)

        # Final projection
        decoder_output = self.final_projection(decoder_output)

        if self.use_lexical_model:

            for j in range(tgt_time_steps):
                lexical_context_vector = lexical_contexts[j]
                transformed_lexical_context = torch.tanh(self.lexical_embedding_projection(lexical_context_vector))
                lexical_vocab_projection = self.lexical_output_projection(transformed_lexical_context)
                decoder_output[:, j, :] += lexical_vocab_projection

        return decoder_output, attn_weights


@register_model_architecture('lstm', 'lstm')
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 64)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', 64)
    args.encoder_num_layers = getattr(args, 'encoder_num_layers', 1)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', 'True')
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', 0.25)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', 0.25)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 64)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 128)
    args.decoder_num_layers = getattr(args, 'decoder_num_layers', 1)
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', 0.25)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', 0.25)
    args.decoder_use_attention = getattr(args, 'decoder_use_attention', 'True')
    args.decoder_use_lexical_model = getattr(args, 'decoder_use_lexical_model', 'False')
