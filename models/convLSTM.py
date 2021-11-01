# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils as utils
# from loadData import loadData
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttConvLSTM(nn.Module):
    def __init__(self, feats_channels=128,
                 feats_dim=28,
                 n_layers=2,
                 hidden_dim=[64, 16],
                 kernel_size=3,
                 time_steps=71,
                 delta_grid_size=15,
                 use_bn=True, predict_delta=False):

        super(AttConvLSTM, self).__init__()
        self.predict_delta = predict_delta
        self.feat_channels = feats_channels
        self.feat_dim = feats_dim
        self.grid_size = feats_dim
        self.delta_grid_size = delta_grid_size
        self.use_bn = use_bn
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.time_steps = time_steps

        assert(len(self.hidden_dim) == self.n_layers)

        # init
        self.conv_x = []  # Conv2d used for feature x
        self.conv_h = []  # Conv2d used for outer hidden state h
        if self.use_bn:
            self.bn_x = []
            self.bn_h = []
            self.bn_c = []

        for l in range(n_layers):
            hidden_dim = self.hidden_dim[l]
            if l != 0:
                in_channels = self.hidden_dim[l-1]
            else:
                in_channels = self.feat_channels + 2


            self.conv_x.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=4 * hidden_dim,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size//2,
                    bias=not self.use_bn
                )
            )

            self.conv_h.append(
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=4*hidden_dim,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size//2,
                    bias=not self.use_bn
                )
            )

            if use_bn:
                self.bn_x.append(nn.ModuleList([nn.BatchNorm2d(4 * hidden_dim) for i in range(time_steps)]))
                self.bn_h.append(nn.ModuleList([nn.BatchNorm2d(4 * hidden_dim) for i in range(time_steps)]))
                self.bn_c.append(nn.ModuleList([nn.BatchNorm2d(hidden_dim) for i in range(time_steps)]))

        self.conv_x = nn.ModuleList(self.conv_x)
        self.conv_h = nn.ModuleList(self.conv_h)
        if self.use_bn:
            self.bn_x = nn.ModuleList(self.bn_x)
            self.bn_h = nn.ModuleList(self.bn_h)
            self.bn_c = nn.ModuleList(self.bn_c)

        self.att_in_planes = sum(self.hidden_dim)

        # used for cal attn-weights, 1*1conv, not fc
        self.conv_att = nn.Conv2d(
            in_channels=self.att_in_planes,
            out_channels=self.feat_channels,
            kernel_size=1,
            padding=0,
            bias=True
        )

        self.fc_att = nn.Linear(
            in_features=self.feat_channels,
            out_features=1
        )

        # RNN output to pred point
        self.fc_out = nn.Linear(
            in_features=self.grid_size**2 * self.hidden_dim[-1],
            out_features=self.grid_size**2 + 1  # +EOS:(28*28+1)
        )

        # delta
        if predict_delta:
            self.delta_downsample_conv = nn.Conv2d(in_channels=17,
                                                   out_channels=17,
                                                   kernel_size=3,
                                                   stride=2,
                                                   padding=1,
                                                   bias=True)
            self.delta_conv1 = nn.Conv2d(in_channels=128+17,
                                         out_channels=64,
                                         kernel_size=3,
                                         padding=1,
                                         stride=1,
                                         bias=True)
            self.delta_bn1 = nn.BatchNorm2d(64)
            self.delta_res1_conv = nn.Conv2d(in_channels=128+17,
                                        out_channels=64,
                                        kernel_size=1,
                                        stride=1,
                                        bias=True)
            self.delta_res1_bn = nn.BatchNorm2d(64)
            self.delta_relu1 = nn.ReLU(inplace=True)  # out: (N, 64, 14, 14)

            self.delta_conv2 = nn.Conv2d(in_channels=64,
                                         out_channels=16,
                                         kernel_size=3,
                                         padding=1,
                                         stride=1,
                                         bias=True)

            self.delta_bn2 = nn.BatchNorm2d(16)
            self.delta_res2_conv = nn.Conv2d(in_channels=64,
                                        out_channels=16,
                                        kernel_size=1,
                                        stride=1,
                                        bias=True)
            self.delta_res2_bn = nn.BatchNorm2d(16)
            self.delta_relu2 = nn.ReLU(inplace=True)

            self.delta_final = nn.Linear(16*14*14, delta_grid_size**2)

            self.use_delta_attn = True
            # delta attn
            self.delta_fc_att = nn.Linear(in_features=self.feat_channels,
                                          out_features=1)
            self.delta_conv_att = nn.Conv2d(in_channels=17,
                                            out_channels=self.feat_channels,
                                            kernel_size=1,
                                            padding=0,
                                            bias=True)

    def delta_step(self, last_rnn_h, rnn_logits, encoder_features, mode='train_ce', temperature=0.0):
        """

        :param last_rnn_h:
        :param rnn_logits:
        :param delta_features:
        :return:
        """
        pre_features = torch.cat([last_rnn_h, rnn_logits.unsqueeze(1)], dim=1)
        pre_features = self.delta_downsample_conv(pre_features)  # [N, 17, 14, 14]

        # Use attn, same with rnn attention mechanism
        if self.use_delta_attn:
            encoder_features = self.delta_attn(pre_features, encoder_features)

        features = torch.cat([pre_features, encoder_features], dim=1)  # [N, 128+17, 14, 14]

        res1 = self.delta_res1_conv(features)
        res1 = self.delta_res1_bn(res1)

        out1 = self.delta_conv1(features)
        out1 = self.delta_bn1(out1)

        out1 = self.delta_relu1(out1 + res1)

        res2 = self.delta_res2_conv(out1)
        res2 = self.delta_res2_bn(res2)
        out2 = self.delta_conv2(out1)
        out2 = self.delta_bn2(out2)

        out = self.delta_relu2(res2 + out2)  # (N, 16, 14, 14)

        bs = out.shape[0]
        out = out.view(bs, -1)
        delta_logits = self.delta_final(out)  # (N, 14*14)

        out_delta = {}

        out_delta['delta_logits'] = delta_logits

        delta_logprobs = F.log_softmax(delta_logits, dim=-1)
        if temperature < 0.01:
            delta_logprob, delta_pred = torch.max(delta_logprobs, dim=-1)  # greedy
            out_delta['delta_logprob'] = delta_logprob
            out_delta['delta_pred'] = delta_pred

        else:
            probs = torch.exp(delta_logprobs / temperature)
            cur_pred = torch.multinomial(probs, 1)
            cur_logprob = delta_logprobs.gather(1, cur_pred)
            cur_pred = torch.squeeze(cur_pred, dim=-1)  # (bs, )
            cur_logprob = torch.squeeze(cur_logprob, dim=-1)  # prob (bs,)
            out_delta['delta_logprob'] = cur_logprob
            out_delta['delta_pred'] = cur_pred

        return out_delta



        # return delta_logits

    def rnn_step(self, t, feats, cur_state):

        out_state = []

        for l in range(self.n_layers):
            h_cur, c_cur = cur_state[l]
            if l == 0:
                input = feats
            else:
                input = out_state[l-1][0]  # previous layer outer hidden

            # forward
            conv_x = self.conv_x[l]
            feats_x = conv_x(input)
            if self.use_bn:

                feats_x = self.bn_x[l][t](feats_x)

            conv_h = self.conv_h[l]
            feats_h = conv_h(h_cur)
            if self.use_bn:
                feats_h = self.bn_h[l][t](feats_h)

            i, f, o, u = torch.split((feats_x + feats_h), self.hidden_dim[l], dim=1)

            c = F.sigmoid(f) * c_cur + F.sigmoid(i) * F.tanh(u)

            if self.use_bn:
                h = F.sigmoid(o) * F.tanh(self.bn_c[l][t](c))
            else:
                h = F.sigmoid(o) * F.tanh(c)
            out_state.append([h,c])

        return out_state

    def rnn_zero_state(self, shape):
        out_state = []
        # shape: (b,c,h,w)
        for l in range(self.n_layers):
            h = torch.zeros(shape[0], self.hidden_dim[l], shape[2], shape[3], device=device)  # (b, hidden_dim, h, w)
            c = torch.zeros(shape[0], self.hidden_dim[l], shape[2], shape[3], device=device)
            out_state.append([h,c])

        return out_state

    def attn(self, feats, pre_state):

        h_cat = torch.cat([state[0] for state in pre_state], dim=1)
        # h_cat : (N, hidden_dim1+hidden_dim2, h, w)
        # f12: (N, 128, h, w)
        f12 = self.conv_att(h_cat)  # paper中的f1,f2
        f12 = F.relu(f12)
        # (N*28*28, 128)
        f12 = f12.permute(0, 2, 3, 1).contiguous().view(-1, self.feat_channels)
        # f_att: (N*28*28,1)
        f_att = self.fc_att(f12)
        # f_att: (N, 28*28)
        f_att = f_att.contiguous().view(-1, self.grid_size**2)
        a_t = F.log_softmax(f_att, dim=-1)  # paper中的Alpha_t
        a_t = a_t.view(-1, 1, self.grid_size, self.grid_size)  # (N, 1, 28, 28)

        feats = feats * a_t  # broadcast
        return feats, a_t

    # delta-attention mechanism
    def delta_attn(self, input, feats):

        f12 = self.delta_conv_att(input)  # [N, 128, 14, 14]
        f12 = F.relu(f12)  # [N, 128, 14, 14]
        f12 = f12.permute(0, 2, 3, 1).contiguous().view(-1, self.feat_channels)  # [N*14*14, 128]
        f_att = self.delta_fc_att(f12)  # [N*14*14, 1]
        f_att = f_att.contiguous().view(-1, (self.delta_grid_size-1) ** 2)  # (N, 14*14)
        a_t = F.log_softmax(f_att, dim=-1)
        a_t = a_t.view(-1, 1, self.delta_grid_size-1, self.delta_grid_size-1)  # (N, 1, 15, 15)
        feats = feats * a_t
        return feats

    def forward(self, feats,
                delta_feats,
                pre_v2,
                pre_v1,
                temperature=0.0,
                mode='train_ce',
                fp_beam_size=1,
                beam_size=1,
                return_attention=False,
                use_correction=False):


        params = locals()
        params.pop('self')
        if beam_size == 1:
            outdict = self.vanilla_forward(**params)
            return outdict
        else:
            return self.beam_forward(**params)

    def vanilla_forward(self,
                        feats,
                        delta_feats,
                        pre_v2,
                        pre_v1,
                        temperature=0.0,
                        mode='train_ce',
                        fp_beam_size=1,
                        beam_size=1,
                        return_attention=False,
                        use_correction=False):
        """

        :param feats:
        :param first_v:
        :param pre_v2:
        :param pre_v1:
        :param temperature:
        :param mode:
        :param fp_beam_size:
        :param beam_size:
        :param return_attention:
        :param use_correction:
        :return: out_dict
        """

        N, feats_channels, grid_size, _ = feats.shape
        out_dict = defaultdict()

        if mode == 'train_ce':
            prev2_padding = torch.zeros((N, 2, grid_size, grid_size), device=device)
            prev1_padding = torch.zeros((N, 1, grid_size, grid_size), device=device)
            pre_v2 = pre_v2[:, :, :-1].view(N, -1, grid_size, grid_size)  # (N, len_s-2, 28, 28)
            pre_v1 = pre_v1[:, :, :-1].view(N, -1, grid_size, grid_size)
            pre_v2 = torch.cat([prev2_padding, pre_v2], dim=1)
            pre_v1 = torch.cat([prev1_padding, pre_v1], dim=1)
            # init hidden
            cur_state = self.rnn_zero_state(feats.shape)
            out_logits = []
            out_lengths = torch.zeros(N, device=device).to(torch.long)
            out_lengths += self.time_steps
            out_pred_polys = []  # (bs, time_step)
            out_pred_logprobs = []  # (bs, time_step)
            out_attn = []
            out_delta_logits = []
            out_delta_probs = []
            out_delta_pred = []
            for t in range(self.time_steps):
                attn_feats = self.attn(feats, cur_state)  # [N, 128, 28, 28]
                cur_prev2 = pre_v2[:, t, :, :].view(N, 1, grid_size, grid_size)
                cur_prev1 = pre_v1[:, t, :, :].view(N, 1, grid_size, grid_size)
                rnn_input = torch.cat([attn_feats[0], cur_prev2, cur_prev1], dim=1)
                # rnn_step
                cur_state = self.rnn_step(t, rnn_input, cur_state)
                cur_h_delta = cur_state[-1][0]  # (N, 16, 28, 28)
                cur_h = cur_h_delta.view(N, -1)
                cur_vertex_logits = self.fc_out(cur_h)

                if self.predict_delta:
                    delta_outdict = self.delta_step(cur_h_delta,
                                                   cur_vertex_logits[:, :-1].view(N, 28, 28),
                                                   delta_feats)
                    # out_delta_logits.append(delta_logits)
                    out_delta_logits.append(delta_outdict['delta_logits'])
                    out_delta_probs.append(delta_outdict['delta_logprob'])
                    out_delta_pred.append(delta_outdict['delta_pred'])


                out_logits.append(cur_vertex_logits)  # logits train for train_ce
                out_attn.append(attn_feats[1])  # attn
                logprobs = F.log_softmax(cur_vertex_logits, dim=-1)
                if temperature < 0.01:
                    cur_logprob, cur_pred = torch.max(logprobs, dim=-1)  # greedy
                else:
                    probs = torch.exp(logprobs / temperature)
                    cur_pred = torch.multinomial(probs, 1)
                    cur_logprob = logprobs.gather(1, cur_pred)
                    cur_pred = torch.squeeze(cur_pred, dim=-1)  # (bs, )
                    cur_logprob = torch.squeeze(cur_logprob, dim=-1)  # prob (bs,)

                # lengths
                for b in range(N):
                    if out_lengths[b] != self.time_steps:
                        continue
                        # prediction has ended
                    if cur_pred[b] == self.grid_size ** 2:
                        # if EOS
                        out_lengths[b] = t + 1
                        # t+1 because we want to keep the EOS
                        # for the loss as well (t goes from 0 to self.time_steps-1)

                out_pred_polys.append(cur_pred.to(torch.float32))  # class
                out_pred_logprobs.append(cur_logprob)  # logprobs of pred class


            out_pred_polys = torch.stack(out_pred_polys)
            out_dict['pred_polys'] = out_pred_polys.permute(1, 0)

            out_pred_logprobs = torch.stack(out_pred_logprobs).permute(1, 0)  # (b, self.time_steps)
            out_dict['rnn_state'] = cur_state

            # logprob sums
            logprob_sums = torch.zeros(N)
            for b in torch.arange(N, dtype=torch.int32):
                p = torch.sum(out_pred_logprobs[b, :out_lengths[b]])
                lp = ((5. + out_lengths[b]) / 6.) ** 0.65
                logprob_sums[b] = p

            out_dict['logprob_sums'] = logprob_sums
            out_dict['feats'] = feats
            out_dict['log_probs'] = out_pred_logprobs
            out_logits = torch.stack(out_logits)  # (self.time_steps, b, self.grid_size**2 + 1)
            out_dict['logits'] = out_logits.permute(1, 0, 2)
            out_dict['lengths'] = out_lengths
            if return_attention:
                out_dict['attention'] = out_attn

            if self.predict_delta:
                delta_logits = torch.stack(out_delta_logits)
                out_dict['delta_logits'] = delta_logits.permute(1, 0, 2)
                delta_logprobs = torch.stack(out_delta_probs)
                out_dict['delta_logprobs'] = delta_logprobs.permute(1, 0)
                delta_pred = torch.stack(out_delta_pred)
                out_dict['delta_pred'] = delta_pred.permute(1, 0).float()

        elif mode == 'train_rl':
            # pre_v2, pre_v1 = None
            v_prev2 = torch.zeros((N, 1, grid_size, grid_size), device=device)
            v_prev1 = torch.zeros((N, 1, grid_size, grid_size), device=device)
            # init hidden state
            cur_state = self.rnn_zero_state(feats.shape)

            out_logits = []
            out_lengths = torch.zeros(N, device=device).to(torch.long)
            out_lengths += self.time_steps
            out_pred_polys = []  # (bs, time_step)
            out_pred_logprobs = []  # (bs, time_step)
            out_attn = []
            out_delta_logits = []
            out_delta_probs = []
            out_delta_pred = []
            for t in range(self.time_steps):
                # attn
                attn_feats = self.attn(feats, cur_state)
                rnn_input = torch.cat([attn_feats[0], v_prev2, v_prev1], dim=1)
                # rnn_step
                cur_state = self.rnn_step(t, rnn_input, cur_state)
                cur_h_delta = cur_state[-1][0]  # (N, 16, 28, 28)
                cur_h = cur_h_delta.view(N, -1)
                cur_vertex_logits = self.fc_out(cur_h)  # .view(N, 1, -1)  # (N, 28*28+1)

                # TODO: predict delta
                if self.predict_delta:
                    delta_outdict = self.delta_step(cur_h_delta,
                                                   cur_vertex_logits[:, :-1].view(N, 28, 28),
                                                   delta_feats)

                    out_delta_logits.append(delta_outdict['delta_logits'])
                    out_delta_probs.append(delta_outdict['delta_logprob'])
                    out_delta_pred.append(delta_outdict['delta_pred'])




                out_logits.append(cur_vertex_logits)
                out_attn.append(attn_feats[1])
                logprobs = F.log_softmax(cur_vertex_logits, dim=-1)

                if temperature < 0.01:
                    cur_logprob, cur_pred = torch.max(logprobs, dim=-1)  # greedy
                else:
                    probs = torch.exp(logprobs / temperature)
                    cur_pred = torch.multinomial(probs, 1)
                    # Get logprob of the sampled vertex
                    cur_logprob = logprobs.gather(1, cur_pred)

                    # Remove the last dimension if not 1
                    cur_pred = torch.squeeze(cur_pred, dim=-1)
                    cur_logprob = torch.squeeze(cur_logprob, dim=-1)

                # lengths for each batch
                for b in range(N):
                    if out_lengths[b] != self.time_steps:
                        continue
                        # prediction has ended
                    if cur_pred[b] == self.grid_size ** 2:
                        # if EOS
                        out_lengths[b] = t + 1
                        # t+1 because we want to keep the EOS
                        # for the loss as well (t goes from 0 to self.time_steps-1)

                # update v_pre2/1
                v_prev2 = v_prev2.copy_(v_prev1)
                v_prev1 = utils.class_to_grid(cur_pred, v_prev1, grid_size)

                out_pred_polys.append(cur_pred.to(torch.float32))
                out_pred_logprobs.append(cur_logprob)

            out_pred_polys = torch.stack(out_pred_polys)
            out_dict['pred_polys'] = out_pred_polys.permute(1, 0)

            out_pred_logprobs = torch.stack(out_pred_logprobs).permute(1, 0)  # (b, self.time_steps)

            # rnn hidden state for training ggan
            out_dict['rnn_state'] = cur_state

            # logprob sums
            logprob_sums = torch.zeros(N)
            for b in torch.arange(N, dtype=torch.int32):
                p = torch.sum(out_pred_logprobs[b, :out_lengths[b]])
                lp = ((5. + out_lengths[b]) / 6.) ** 0.65
                logprob_sums[b] = p

            out_dict['logprob_sums'] = logprob_sums
            out_dict['feats'] = feats
            out_dict['log_probs'] = out_pred_logprobs
            out_logits = torch.stack(out_logits)  # (self.time_steps, b, self.grid_size**2 + 1)
            out_dict['logits'] = out_logits.permute(1, 0, 2)
            out_dict['lengths'] = out_lengths
            if return_attention:
                out_dict['attention'] = out_attn

            if self.predict_delta:
                delta_logits = torch.stack(out_delta_logits)
                out_dict['delta_logits'] = delta_logits.permute(1, 0, 2)
                delta_logprobs = torch.stack(out_delta_probs)
                out_dict['delta_logprobs'] = delta_logprobs.permute(1, 0)
                delta_pred = torch.stack(out_delta_pred)
                out_dict['delta_pred'] = delta_pred.permute(1, 0).float()


        elif mode == 'test':
            # pre_v2, pre_v1 = None
            v_prev2 = torch.zeros((N, 1, grid_size, grid_size), device=device)
            v_prev1 = torch.zeros((N, 1, grid_size, grid_size), device=device)
            # init hidden state
            cur_state = self.rnn_zero_state(feats.shape)

            out_pred_polys = []  # (bs, time_step)
            out_pred_logprobs = []  # (bs, time_step)
            out_lengths = torch.zeros(N, device=device).to(torch.long)
            out_lengths += self.time_steps
            out_delta_logits = []
            out_delta_probs = []
            out_delta_pred = []
            for t in range(self.time_steps):
                # attn
                attn_feats = self.attn(feats, cur_state)
                rnn_input = torch.cat([attn_feats[0], v_prev2, v_prev1], dim=1)
                # rnn_step
                cur_state = self.rnn_step(t, rnn_input, cur_state)
                cur_h_delta = cur_state[-1][0]  # (N, 16, 28, 28)
                cur_h = cur_h_delta.view(N, -1)
                cur_vertex_logits = self.fc_out(cur_h)  # .view(N, 1, -1)  # (N, 28*28+1)

                # predict delta
                if self.predict_delta:
                    delta_outdict = self.delta_step(cur_h_delta,
                                                   cur_vertex_logits[:, :-1].view(N, 28, 28),
                                                   delta_feats)

                    out_delta_logits.append(delta_outdict['delta_logits'])
                    out_delta_probs.append(delta_outdict['delta_logprob'])
                    out_delta_pred.append(delta_outdict['delta_pred'])


                logprobs = F.log_softmax(cur_vertex_logits, dim=-1)
                if temperature < 0.01:
                    cur_logprob, cur_pred = torch.max(logprobs, dim=-1)  # greedy
                else:
                    probs = torch.exp(logprobs / temperature)
                    cur_pred = torch.multinomial(probs, 1)
                    # Get logprob of the sampled vertex
                    cur_logprob = logprobs.gather(1, cur_pred)

                    # Remove the last dimension if not 1
                    cur_pred = torch.squeeze(cur_pred, dim=-1)
                    cur_logprob = torch.squeeze(cur_logprob, dim=-1)

                 # lengths for each batch
                for b in range(N):
                    if out_lengths[b] != self.time_steps:
                        continue
                        # prediction has ended
                    if cur_pred[b] == self.grid_size ** 2:
                        # if EOS
                        out_lengths[b] = t + 1
                        # t+1 because we want to keep the EOS
                        # for the loss as well (t goes from 0 to self.time_steps-1)

                v_prev2 = v_prev2.copy_(v_prev1)
                v_prev1 = utils.class_to_grid(cur_pred, v_prev1, grid_size)

                out_pred_polys.append(cur_pred.to(torch.float32))
                out_pred_logprobs.append(cur_logprob)

            # rnn hidden state for training ggan
            out_dict['rnn_state'] = cur_state
            out_dict['lengths'] = out_lengths
            out_pred_polys = torch.stack(out_pred_polys)
            out_dict['pred_polys'] = out_pred_polys.permute(1, 0)
            out_pred_logprobs = torch.stack(out_pred_logprobs).permute(1, 0)  # (b, self.time_steps)
            out_dict['log_probs'] = out_pred_logprobs

            if self.predict_delta:
                delta_logits = torch.stack(out_delta_logits)
                out_dict['delta_logits'] = delta_logits.permute(1, 0, 2)
                delta_logprobs = torch.stack(out_delta_probs)
                out_dict['delta_logprobs'] = delta_logprobs.permute(1, 0)
                delta_pred = torch.stack(out_delta_pred)
                out_dict['delta_pred'] = delta_pred.permute(1, 0).float()

        return out_dict

    def beam_forward(self,
                     feats,
                     delta_feats,
                     pre_v2,
                     pre_v1,
                     temperature=0.0,
                     mode='train_ce',
                     fp_beam_size=1,
                     beam_size=1,
                     return_attention=False,
                     use_correction=False):


        batch_size = feats.size(0)
        fp_beam_batch_size = batch_size * fp_beam_size
        full_beam_batch_size = fp_beam_batch_size * beam_size


        v_prev2 = torch.zeros(fp_beam_batch_size, 1, self.grid_size, self.grid_size, device=device)
        v_prev1 = torch.zeros(fp_beam_batch_size, 1, self.grid_size, self.grid_size, device=device)

        tokens_table = torch.ones(fp_beam_batch_size, beam_size, self.time_steps, device=device)
        tokens_table = tokens_table * self.grid_size ** 2  # Fill with EOS token

        logprob_sums = torch.zeros((fp_beam_batch_size, beam_size)).to(device)
        alive = torch.ones(fp_beam_batch_size, beam_size, 1, device=device)
        # A vector of beams that are alive of size [fp_beam_batch_size, beam_size, 1]

        for t in range(0, self.time_steps):
            # 第一步
            if t == 0:
                if fp_beam_size > 1:
                    feats = feats.unsqueeze(1)
                    feats = feats.repeat([1, fp_beam_size, 1, 1, 1])
                    feats = feats.view(fp_beam_batch_size, -1, self.grid_size, self.grid_size)

                rnn_state = self.rnn_zero_state(feats.size())

                att_feats, att = self.attn(feats, rnn_state)
                input_t = torch.cat((att_feats, v_prev2, v_prev1), dim=1)

                rnn_state = self.rnn_step(t, input_t, rnn_state)

                h_final = rnn_state[-1][0]
                h_final = h_final.view(fp_beam_batch_size, -1)

                logits_t = self.fc_out(h_final)
                logprob = F.log_softmax(logits_t, dim=-1)

                val, idx = torch.topk(logprob, beam_size, dim=-1)

                # Update alive
                alive = idx.ne(self.grid_size ** 2).unsqueeze(2).float()

                logprob_sums += val

                tokens_table[:, :, t] = idx

                new_rnn_state = []
                for l in torch.arange(self.n_layers, dtype=torch.int32):
                    h = rnn_state[l][0]
                    c = rnn_state[l][1]

                    h = h.unsqueeze(1).repeat(1, beam_size, 1, 1, 1)
                    h = h.view(full_beam_batch_size, -1, self.grid_size, self.grid_size)
                    c = c.unsqueeze(1).repeat(1, beam_size, 1, 1, 1)
                    c = c.view(full_beam_batch_size, -1, self.grid_size, self.grid_size)

                    new_rnn_state.append([h, c])

                rnn_state = new_rnn_state


                v_prev2 = torch.zeros(full_beam_batch_size, 1, self.grid_size, self.grid_size, device=device)
                v_prev1 = torch.zeros(full_beam_batch_size, 1, self.grid_size, self.grid_size, device=device)
                v_first = torch.zeros(full_beam_batch_size, 1, self.grid_size, self.grid_size, device=device)


                v_prev1 = utils.class_to_grid(tokens_table[:, :, t].view(-1), v_prev1, self.grid_size)
                v_prev2 = utils.class_to_grid(tokens_table[:, :, t - 1].view(-1), v_prev2, self.grid_size)

                feats = feats.unsqueeze(1)
                feats = feats.repeat([1, beam_size, 1, 1, 1])
                feats = feats.view(full_beam_batch_size, -1, self.grid_size, self.grid_size)
            # 其他步
            else:
                # For t>=1
                att_feats, att = self.attn(feats, rnn_state)
                input_t = torch.cat((att_feats, v_prev2, v_prev1), dim=1)

                rnn_state = self.rnn_step(t, input_t, rnn_state)

                h_final = rnn_state[-1][0]  # h from last layer
                h_final = h_final.view(full_beam_batch_size, -1)

                logits_t = self.fc_out(h_final)
                logprob = F.log_softmax(logits_t, dim=-1)
                # shape = [full_beam_batch_size, self.grid_size**2 + 1]

                logprob = logprob.view(fp_beam_batch_size, beam_size, -1)

                logprob = logprob * alive  # / (t+1)
                # If alive, then length penalty for current logprob
                # alive is of shape [fp_beam_batch_size, beam_size, 1]

                logprob_sums = logprob_sums.unsqueeze(2)  # * (torch.abs(alive-1) + alive*t/(t+1))
                # If alive, then scale previous logprob_sums down else keep same value
                # This is of shape [fp_beam_batch_size, beam_size, 1]

                logprob = logprob + logprob_sums
                # [fp_beam_batch_size, beam_size, self.grid_size**2 + 1]

                lengths = torch.sum(tokens_table.ne(self.grid_size ** 2).float(), dim=-1).unsqueeze(-1)
                # [fp_beam_batch_size, beam_size, 1]

                lp = ((5. + lengths) / 6.) ** 0.65
                logprob_pen = logprob / lp

                # For those sequences that have ended, we mask out all the logprobs
                # of the locations except the EOS token, to avoid double counting while sorting
                mask = torch.eq(alive, 0).repeat(1, 1, self.grid_size ** 2 + 1)
                mask[:, :, -1] = 0
                # keep the EOS token alive for all beams

                min_val = torch.min(logprob_pen) - 1
                logprob_pen.masked_fill_(mask, min_val)
                # Fill ended sequences with the minimum logprob - 1, except at the EOS token

                logprob_pen = logprob_pen.view(fp_beam_batch_size, -1)  # [fp_beam_batch_size, beam_size * num_tokens]
                logprob = logprob.view(fp_beam_batch_size, -1)  # [fp_beam_batch_size, beam_size * num_tokens]
                val, idx = torch.topk(logprob_pen, beam_size, dim=-1)

                logprob_sums = logprob.gather(1, idx)
                # [fp_beam_batch_size, beam_size]

                beam_idx = idx / (self.grid_size ** 2 + 1)
                token_idx = idx % (self.grid_size ** 2 + 1)

                # Update tokens table
                for b in torch.arange(fp_beam_batch_size, dtype=torch.int32):
                    beams_to_keep = tokens_table[b, :, :].index_select(0, beam_idx[b, :])
                    # This is [beam_size, time_steps]
                    beams_to_keep[:, t] = token_idx[b, :]
                    # Add current prediction to the beams

                    tokens_table[b, :, :] = beams_to_keep

                # Update hidden state
                new_rnn_state = []
                for l in torch.arange(self.n_layers, dtype=torch.int32):
                    h = rnn_state[l][0]
                    c = rnn_state[l][1]
                    # Both are of shape [batch_size*beam_size, self.hidden_size[l], self.grid_size, self.grid_size]

                    h = h.view(fp_beam_batch_size, beam_size, -1, self.grid_size, self.grid_size)
                    c = c.view(fp_beam_batch_size, beam_size, -1, self.grid_size, self.grid_size)

                    for b in torch.arange(fp_beam_batch_size, dtype=torch.int32):
                        h[b, ...] = h[b, ...].index_select(0, beam_idx[b, :])
                        c[b, ...] = c[b, ...].index_select(0, beam_idx[b, :])

                    h = h.view(full_beam_batch_size, -1, self.grid_size, self.grid_size)
                    c = c.view(full_beam_batch_size, -1, self.grid_size, self.grid_size)

                    new_rnn_state.append([h, c])

                rnn_state = new_rnn_state

                # Update v_prev2, v_prev1 and v_first
                v_prev2 = v_prev2.copy_(v_prev1)
                v_prev1 = utils.class_to_grid(tokens_table[:, :, t].view(-1), v_prev1, self.grid_size)
                # v_first = utils.class_to_grid(tokens_table[:, :, 0].view(-1), v_first, self.grid_size)

                # Update alive vector
                alive = torch.ne(tokens_table[:, :, t], self.grid_size ** 2).float()
                alive = alive.unsqueeze(2)
                # This works because if a beam was not alive and it was selected again
                # then the token that was selected has to be the EOS token because
                # we masked out the logprobs at the other tokens and gave them the min value

                # print alive[:,:,0]
                # print logprob_sums

        tokens_table = tokens_table.view(batch_size, fp_beam_size, beam_size, -1)
        logprob_sums = logprob_sums.view(batch_size, fp_beam_size, beam_size)

        out_dict = {}
        out_dict['feats'] = feats
        # Return the reshape feats based on beam sizes

        out_dict['logprob_sums'] = logprob_sums.view(full_beam_batch_size)
        out_dict['pred_polys'] = tokens_table.view(full_beam_batch_size, -1)
        out_dict['rnn_state'] = rnn_state
        # Return the last rnn state

        return out_dict


if __name__ == '__main__':
    model = AttConvLSTM(feats_channels=128, feats_dim=28, predict_delta=True)  # (Seqlen,C,W,H)
    for n, p in model.named_parameters():
        print(n)
    # print(model)
    # print [c for c in model.children()]
    # dataloader = loadData('train', 16, 70, 16)
    # for i,batch in enumerate(dataloader):
    #     if i > 1:
    #         break
    # pre_v2 = torch.zeros((2, 69, 28*28+1))
    # pre_v1 = torch.zeros((2, 70, 28*28+1))
    # feats = torch.rand((2, 128, 28, 28))
    # model.eval()
    # x = model(feats, pre_v2, pre_v1, mode='train_rl', temperature=0.6, beam_size=4)
    # # y = model(feats, pre_v2, pre_v1, mode='train_rl', temperature=0.0)
    # # print(x['pred_polys'])
    # # print(y['pred_polys'])
    # # print(x)
    #
    # for key in x:
    #      print(key, type(x[key]))
    #      if isinstance(x[key], torch.Tensor):
    #          print(key, x[key].shape)
    #      else:
    #          print(key, len(x[key]))

