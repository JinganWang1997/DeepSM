from modules import *
import torch
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
ids = [0, 1, 2, 3]


class DeepSM_ResUNet(nn.Module):

    def __init__(self, input_c, output_c=1, channel_num=64):
        super(DeepSM_ResUNet, self).__init__()

        self.in_dim = input_c
        self.out_dim = channel_num
        self.final_out_dim = output_c

        self.down_1 = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_dim),
            nn.ReLU(),
            nn.Conv2d(self.out_dim, self.out_dim, kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, kernel_size=3, padding=1)
        )

        self.pool_1 = maxpool()

        self.down_2 = ResConvBlock2d(self.out_dim * 1, self.out_dim * 2)
        self.pool_2 = maxpool()
        self.down_3 = ResConvBlock2d(self.out_dim * 2, self.out_dim * 4)
        self.pool_3 = maxpool()
        self.down_4 = ResConvBlock2d(self.out_dim * 4, self.out_dim * 8)
        self.pool_4 = maxpool()


        self.bridge = ResConvBlock2d(self.out_dim * 8, self.out_dim * 16)

        self.upLayer1 = UpBlockSingleConv2d(self.out_dim * 16, self.out_dim * 8)
        self.upLayer1_ResConv = ResConvBlock2d(self.out_dim * 16, self.out_dim * 8)
        self.upLayer2 = UpBlockSingleConv2d(self.out_dim * 8, self.out_dim * 4)
        self.upLayer2_ResConv = ResConvBlock2d(self.out_dim * 8, self.out_dim * 4)
        self.upLayer3 = UpBlockSingleConv2d(self.out_dim * 4, self.out_dim * 2)
        self.upLayer3_ResConv = ResConvBlock2d(self.out_dim * 4,  self.out_dim * 2)
        self.upLayer4 = UpBlockSingleConv2d(self.out_dim * 2, self.out_dim * 1)
        self.upLayer4_ResConv = ResConvBlock2d(self.out_dim * 2, self.out_dim * 1)

    def forward(self, input):

        i0 = input[:, :, :, :]

        down_layer_1 = self.down_1(i0) + self.input_skip(i0)
        down_layer_1_m = self.pool_1(down_layer_1)

        down_layer_2 = self.down_2(down_layer_1_m)
        down_layer_2_m = self.pool_2(down_layer_2)


        down_layer_3 = self.down_3(down_layer_2_m)
        down_layer_3_m = self.pool_3(down_layer_3)

        down_layer_4 = self.down_4(down_layer_3_m)
        down_layer_4_m = self.pool_4(down_layer_4)

        bridge = self.bridge(down_layer_4_m)

        skip_1 = down_layer_4
        skip_2 = down_layer_3
        skip_3 = down_layer_2
        skip_4 = down_layer_1

        up_1 = self.upLayer1(bridge, skip_1)
        up_1_skip = self.upLayer1_ResConv(up_1)
        up_2 = self.upLayer2(up_1_skip, skip_2)
        up_2_skip = self.upLayer2_ResConv(up_2)
        up_3 = self.upLayer3(up_2_skip, skip_3)
        up_3_skip = self.upLayer3_ResConv(up_3)
        up_4 = self.upLayer4(up_3_skip, skip_4)
        up_4_skip = self.upLayer4_ResConv(up_4)
        x = up_4_skip

        return x




class BiConvLSTM0(nn.Module):
    def __init__(self, in_c=1, channel_num=64):
        super(BiConvLSTM0, self).__init__()
        self.conv_gx_lstm0 = nn.Conv2d(in_c + channel_num, channel_num, kernel_size=3, padding=1)
        self.conv_ix_lstm0 = nn.Conv2d(in_c + channel_num, channel_num, kernel_size=3, padding=1)
        self.conv_ox_lstm0 = nn.Conv2d(in_c + channel_num, channel_num, kernel_size=3, padding=1)

    def forward(self, xt):

        gx = self.conv_gx_lstm0(xt)
        ix = self.conv_ix_lstm0(xt)
        ox = self.conv_ox_lstm0(xt)

        gx = torch.tanh(gx)
        ix = torch.sigmoid(ix)
        ox = torch.sigmoid(ox)

        cell_1 = torch.tanh(gx * ix)
        hide_1 = ox * cell_1
        return cell_1, hide_1



class BiConvLSTM(nn.Module):
    def __init__(self, in_c=1, channel_num=64):
        super(BiConvLSTM, self).__init__()
        self.conv_ix_lstm = nn.Conv2d(in_c + channel_num, channel_num, kernel_size=3, padding=1, bias=True)
        self.conv_ih_lstm = nn.Conv2d(channel_num, channel_num, kernel_size=3, padding=1, bias=False)

        self.conv_fx_lstm = nn.Conv2d(in_c + channel_num, channel_num, kernel_size=3, padding=1, bias=True)
        self.conv_fh_lstm = nn.Conv2d(channel_num, channel_num, kernel_size=3, padding=1, bias=False)

        self.conv_ox_lstm = nn.Conv2d(in_c + channel_num, channel_num, kernel_size=3, padding=1, bias=True)
        self.conv_oh_lstm = nn.Conv2d(channel_num, channel_num, kernel_size=3, padding=1, bias=False)

        self.conv_gx_lstm = nn.Conv2d(in_c + channel_num, channel_num, kernel_size=3, padding=1, bias=True)
        self.conv_gh_lstm = nn.Conv2d(channel_num, channel_num, kernel_size=3, padding=1, bias=False)

    def forward(self, xt, cell_t_1, hide_t_1):

        gx = self.conv_gx_lstm(xt)
        gh = self.conv_gh_lstm(hide_t_1)
        g_sum = gx + gh
        gt = torch.tanh(g_sum)

        ox = self.conv_ox_lstm(xt)
        oh = self.conv_oh_lstm(hide_t_1)
        o_sum = ox + oh
        ot = torch.sigmoid(o_sum)

        ix = self.conv_ix_lstm(xt)
        ih = self.conv_ih_lstm(hide_t_1)
        i_sum = ix + ih
        it = torch.sigmoid(i_sum)

        fx = self.conv_fx_lstm(xt)
        fh = self.conv_fh_lstm(hide_t_1)
        f_sum = fx + fh
        ft = torch.sigmoid(f_sum)

        cell_t = ft * cell_t_1 + it * gt
        hide_t = ot * torch.tanh(cell_t)

        return cell_t, hide_t


class Deep_SM(nn.Module):
    def __init__(self, input_c=1, output_c=1, channel_num=64, temporal=1000):
        super(Deep_SM, self).__init__()
        self.temporal = temporal
        self.resunet = DeepSM_ResUNet(input_c, output_c, channel_num)
        self.biconvlstm0 = BiConvLSTM0(in_c=output_c , channel_num=channel_num)
        self.biconvlstm = BiConvLSTM(in_c=output_c , channel_num=channel_num)

        self.resunet_out = nn.Conv2d(channel_num, output_c, kernel_size=3, stride=1, padding=1)
        self.out = nn.Conv2d(channel_num, output_c, kernel_size=3, stride=1, padding=1)
        self.out_bilstm = nn.Conv2d(2*channel_num, output_c, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        DeepSM_output = []
        resunet_output = []
        cell = None
        hide = None

        lstm_in_set =[]
        hide_combined_1 = []

        Frame_num = x.size(1)
        for t in range(Frame_num):
            im_t = x[:, t, :, :, :]
            resunet_last = self.resunet(im_t)
            resunet_out_t = self.resunet_out(resunet_last)
            resunet_output.append(resunet_out_t)
            lstm_in = torch.cat((resunet_out_t, resunet_last), dim=1)
            lstm_in_set.append(lstm_in)

            if t == 0:
                cell, hide = self.biconvlstm0(lstm_in)
            else:
                cell, hide = self.biconvlstm(lstm_in, cell, hide)


            hide_combined_1.append(hide)

        hide_combined_2 = []
        for t in range(Frame_num):

            if t == Frame_num-1:
                cell, hide = self.biconvlstm0(lstm_in_set[t])
            else:
                cell, hide = self.biconvlstm(lstm_in_set[t], cell, hide)

            hide_combined_2.append(hide)

        for t in range(Frame_num):
            hide_1 = hide_combined_1[t]
            hide_2 = hide_combined_2[t]
            hide_combined = torch.cat((hide_1, hide_2), dim=1)
            out_lstm_t = self.out_bilstm (hide_combined)
            DeepSM_output.append(out_lstm_t)

        return torch.stack(resunet_output, dim=1), torch.stack(DeepSM_output, dim=1)


if __name__ == "__main__":
    channel_num = 64


