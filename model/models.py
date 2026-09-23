import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl


def model_PU(x, model, record=False):
    if record:
        y, logits_m = model.sample_m(x, m=8, testing=True)
        return y, logits_m
    else:
        y = model.sample_m(x, m=8, testing=True)
        return y

def Uentropy(logits, c):
    pc = F.softmax(logits, dim=1)
    logpc = F.log_softmax(logits, dim=1)
    u_all = -pc * logpc / math.log(c)
    NU = torch.sum(u_all[:, 1:u_all.shape[1], ...], dim=1)
    return NU

def PU_uncertainty(logits_m):
    p = torch.softmax(logits_m, dim=2)
    p_mean = p.mean(dim=1)
    u = -torch.sum(p_mean * torch.log(p_mean + 1e-8), dim=1, keepdim=True)
    u = u / math.log(p_mean.shape[1])
    return u

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    
def init_weights(m):
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        truncated_normal_(m.bias, mean=0, std=0.001)

def init_weights_orthogonal_normal(m):
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
     
class DownConvBlock3D(nn.Module):
    def __init__(self, input_dim, output_dim, initializers, padding, pool=True, norm=False):
        super(DownConvBlock3D, self).__init__()
        layers = []
        if pool:
            layers.append(nn.AvgPool3d(kernel_size=2, stride=2, padding=0, ceil_mode=True))
        layers.append(nn.Conv3d(input_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv3d(output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv3d(output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))

        if norm:
            layers.append(nn.BatchNorm3d(output_dim))
        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, patch):
        return self.layers(patch)

class UpConvBlock3D(nn.Module):
    def __init__(self, input_dim, output_dim, initializers, padding, bilinear=True, norm=False):
        super(UpConvBlock3D, self).__init__()
        self.bilinear = bilinear
        if not self.bilinear:
            self.upconv_layer = nn.ConvTranspose3d(input_dim, output_dim, kernel_size=2, stride=2)
            self.upconv_layer.apply(init_weights)
        self.conv_block = DownConvBlock3D(input_dim, output_dim, initializers, padding, pool=False, norm=norm)

    def forward(self, x, bridge):
        if self.bilinear:
            up = nn.functional.interpolate(x, mode='trilinear', scale_factor=2, align_corners=True)
        else:
            up = self.upconv_layer(x)
        assert up.shape[4] == bridge.shape[4]
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out
    
class Encoder3D(nn.Module):
    def __init__(self, input_channels, num_filters, no_convs_per_block, initializers, num_classes, padding=True, posterior=False):
        super(Encoder3D, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters

        if posterior:
            self.input_channels += num_classes

        layers = []
        for i in range(len(self.num_filters)):
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]

            if i != 0:
                layers.append(nn.AvgPool3d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

            layers.append(nn.Conv3d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block - 1):
                layers.append(nn.Conv3d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)
        self.layers.apply(init_weights)

    def forward(self, input):
        output = self.layers(input)
        return output

class AxisAlignedConvGaussian3D(nn.Module):
    def __init__(self, input_channels, num_filters, no_convs_per_block, latent_dim, initializers, num_classes, posterior=False):
        super(AxisAlignedConvGaussian3D, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        self.encoder = Encoder3D(self.input_channels, self.num_filters, self.no_convs_per_block, initializers,
                                 num_classes, posterior=self.posterior)
        self.conv_layer = nn.Conv3d(num_filters[-1], 2 * self.latent_dim, (1, 1, 1), stride=1)
        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer.bias)

    def forward(self, input, segm=None):
        if segm is not None:
            input = torch.cat((input, segm), dim=1)
        encoding = self.encoder(input)
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)
        encoding = torch.mean(encoding, dim=4, keepdim=True)
        mu_log_sigma = self.conv_layer(encoding)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu = mu_log_sigma[:, :self.latent_dim]
        log_sigma = mu_log_sigma[:, self.latent_dim:]
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)
        assert torch.isfinite(mu).all(), "mu has NaN"
        assert torch.isfinite(log_sigma).all(), "log_sigma has NaN"
        return dist

class Fcomb3D(nn.Module):
    def __init__(self, num_filters, latent_dim, num_output_channels, num_classes, no_convs_fcomb, initializers, use_tile=True):
        super(Fcomb3D, self).__init__()
        self.num_channels = num_output_channels
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [2, 3, 4]
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.no_convs_fcomb = no_convs_fcomb

        if self.use_tile:
            layers = []
            layers.append(nn.Conv3d(self.num_filters[0] + self.latent_dim, self.num_filters[0], kernel_size=1))
            layers.append(nn.ReLU(inplace=True))
            for _ in range(no_convs_fcomb - 2):
                layers.append(nn.Conv3d(self.num_filters[0], self.num_filters[0], kernel_size=1))
                layers.append(nn.ReLU(inplace=True))
            self.layers = nn.Sequential(*layers)
            self.last_layer = nn.Conv3d(self.num_filters[0], self.num_classes, kernel_size=1)
            if initializers['w'] == 'orthogonal':
                self.layers.apply(init_weights_orthogonal_normal)
                self.last_layer.apply(init_weights_orthogonal_normal)
            else:
                self.layers.apply(init_weights)
                self.last_layer.apply(init_weights)

    def tile(self, a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*repeat_idx)
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(a.device)
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z):
        if self.use_tile:
            z = torch.unsqueeze(z, 2)
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])
            z = torch.unsqueeze(z, 3)
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])
            z = torch.unsqueeze(z, 4)
            z = self.tile(z, 4, feature_map.shape[self.spatial_axes[2]])
            feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
            output = self.layers(feature_map)
            return self.last_layer(output)
        
class Unet3D_PU(nn.Module):
    def __init__(self, input_channels, num_classes, num_filters, initializers, apply_last_layer=True, padding=True, norm=False):
        super(Unet3D_PU, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.padding = padding
        self.activation_maps = []
        self.apply_last_layer = apply_last_layer
        self.contracting_path = nn.ModuleList()

        for i in range(len(self.num_filters)):
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = self.num_filters[i]

            if i == 0:
                pool = False
            else:
                pool = True
            self.contracting_path.append(
                DownConvBlock3D(input_dim, output_dim, initializers, padding, pool=pool, norm=norm)
            )

        self.upsampling_path = nn.ModuleList()
        n = len(self.num_filters) - 2
        for i in range(n, -1, -1):
            input_dim = output_dim + self.num_filters[i]
            output_dim = self.num_filters[i]
            if i == 0:
                norm = False
            self.upsampling_path.append(
                UpConvBlock3D(input_dim, output_dim, initializers, padding, norm=norm)
            )

        if self.apply_last_layer:
            self.last_layer = nn.Conv3d(output_dim, num_classes, kernel_size=1)
            #nn.init.kaiming_normal_(self.last_layer.weight, mode='fan_in', nonlinearity='relu')
            #nn.init.normal_(self.last_layer.bias)

    def forward(self, x, val):
        blocks = []
        for i, down in enumerate(self.contracting_path):
            x = down(x)
            if i != len(self.contracting_path) - 1:
                blocks.append(x)
        for i, up in enumerate(self.upsampling_path):
            x = up(x, blocks[-i-1])
        del blocks
        # Used for saving the activations and plotting
        if val:
            self.activation_maps.append(x)
        if self.apply_last_layer:
            x = self.last_layer(x)

        return x
    
class ProbabilisticUnet3D(nn.Module):
    def __init__(self, input_channels=1, num_classes=1, num_filters=[16, 32, 64, 128], latent_dim=6, no_convs_fcomb=4,
                 beta=10.0, record=False):
        super(ProbabilisticUnet3D, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.initializers = {'w': 'he_normal', 'b': 'normal'}
        self.beta = beta
        self.z_prior_sample = 0

        self.unet = Unet3D_PU(self.input_channels, self.num_classes, self.num_filters, self.initializers,
                           apply_last_layer=False, padding=True).to(self.device)
        self.prior = AxisAlignedConvGaussian3D(self.input_channels, self.num_filters, self.no_convs_per_block,
                                               self.latent_dim, self.initializers, num_classes).to(self.device)
        self.posterior = AxisAlignedConvGaussian3D(self.input_channels, self.num_filters, self.no_convs_per_block,
                                                   self.latent_dim, self.initializers, num_classes, posterior=True).to(self.device)
        self.fcomb = Fcomb3D(self.num_filters, self.latent_dim, self.input_channels, self.num_classes,
                             self.no_convs_fcomb, {'w': 'orthogonal', 'b': 'normal'}, use_tile=True).to(self.device)
        self.record = record

    def forward(self, patch, segm=None, training=True):
        if training:
            self.posterior_latent_space = self.posterior.forward(patch, segm)
        self.prior_latent_space = self.prior.forward(patch)
        self.unet_features = self.unet.forward(patch, False)

    def sample(self, img, testing=False):
        if not testing:
            z_prior = self.prior_latent_space.rsample()
            self.z_prior_sample = z_prior
        else:
            self.forward(img, training=False)
            z_prior = self.prior_latent_space.sample()
            self.z_prior_sample = z_prior
        return self.fcomb.forward(self.unet_features, z_prior)

    def sample_m(self, img, m, testing=False):
        if not testing:
            z_prior = self.prior_latent_space.rsample()
            self.z_prior_sample = z_prior
        else:
            img_num, _, d, h, w = img.shape
            res = torch.ones(img_num, m, self.num_classes, d, h, w).to(img.device)
            self.forward(img, training=False)
            for i in range(m):
                z_prior = self.prior_latent_space.sample()
                self.z_prior_sample = z_prior
                res[:, i] = self.fcomb.forward(self.unet_features, z_prior)
            total_res = torch.sum(res, 1, keepdim=True)
            total_res = torch.squeeze(total_res)
            if self.record:
                return total_res / m, res
            else:
                return total_res / m

    def reconstruct(self, use_posterior_mean=False, calculate_posterior=False, z_posterior=None):
        if use_posterior_mean:
            z_posterior = self.posterior_latent_space.loc
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
        return self.fcomb.forward(self.unet_features, z_posterior)

    def kl_divergence(self, analytic=True, calculate_posterior=False, z_posterior=None):
        if analytic:
            kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        return kl_div

    def elbo(self, segm, analytic_kl=True, reconstruct_posterior_mean=False):
        criterion = nn.BCEWithLogitsLoss(size_average=False, reduce=False, reduction=None)
        z_posterior = self.posterior_latent_space.rsample()
        self.kl = torch.mean(
            self.kl_divergence(analytic=analytic_kl, calculate_posterior=False, z_posterior=z_posterior))
        self.reconstruction = self.reconstruct(use_posterior_mean=reconstruct_posterior_mean, calculate_posterior=False,
                                               z_posterior=z_posterior)
        reconstruction_loss = criterion(input=self.reconstruction, target=segm)
        self.reconstruction_loss = torch.sum(reconstruction_loss)
        self.mean_reconstruction_loss = torch.mean(reconstruction_loss)
        return -(self.mean_reconstruction_loss + self.beta * self.kl)

class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class Udrop3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Udrop3D, self).__init__()

        # Encoder
        self.conv1 = DoubleConv3D(in_channels, 16)
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = DoubleConv3D(16, 32)
        self.pool2 = nn.MaxPool3d(2)
        self.conv3 = DoubleConv3D(32, 64)
        self.pool3 = nn.MaxPool3d(2)
        self.dropoutd1 = nn.Dropout(p=0.5)
        self.conv4 = DoubleConv3D(64, 128)
        self.pool4 = nn.MaxPool3d(2)
        self.dropoutd2 = nn.Dropout(p=0.5)
        self.conv5 = DoubleConv3D(128, 256)

        # Decoder
        self.dropoutu1 = nn.Dropout(p=0.5)
        self.up6 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.conv6 = DoubleConv3D(256, 128)
        self.dropoutu2 = nn.Dropout(p=0.5)
        self.up7 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.conv7 = DoubleConv3D(128, 64)
        self.up8 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.conv8 = DoubleConv3D(64, 32)
        self.up9 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.conv9 = DoubleConv3D(32, 16)
        self.conv10 = nn.Conv3d(16, out_channels, 1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        p3_dropout = self.dropoutd1(p3)
        c4 = self.conv4(p3_dropout)
        p4 = self.pool4(c4)
        p4_dropout = self.dropoutd2(p4)
        c5 = self.conv5(p4_dropout)
        
        # Decoder
        up_6 = self.up6(c5)
        up_6_dropout = self.dropoutu1(up_6)
        merge6 = self.offsetCat(c4, up_6_dropout)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        up_7_dropout = self.dropoutu1(up_7)
        merge7 = self.offsetCat(c3, up_7_dropout)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        # out = self.softmax(c10)
        return c10
    
    def offsetCat(self, inputs, down_outputs):
        outputs = down_outputs

        # depth
        offset_d = abs(inputs.size(2) - outputs.size(2))
        if offset_d > 0:
            if outputs.device.type == 'cuda':
                addition = torch.rand(
                    outputs.size(0),
                    outputs.size(1),
                    offset_d,
                    outputs.size(3),
                    outputs.size(4)).cuda()
            else:
                addition = torch.rand(
                    outputs.size(0),
                    outputs.size(1),
                    offset_d,
                    outputs.size(3),
                    outputs.size(4))
            outputs = torch.cat([outputs, addition], dim=2)

        # height
        offset_h = abs(inputs.size(3) - outputs.size(3))
        if offset_h > 0:
            if outputs.device.type == 'cuda':
                addition = torch.rand(
                    outputs.size(0),
                    outputs.size(1),
                    outputs.size(2),
                    offset_h,
                    outputs.size(4)).cuda()
            else:
                addition = torch.rand(
                    outputs.size(0),
                    outputs.size(1),
                    outputs.size(2),
                    offset_h,
                    outputs.size(4))
            outputs = torch.cat([outputs, addition], dim=3)

        # width
        offset_w = abs(inputs.size(4) - outputs.size(4))
        if offset_w > 0:
            if outputs.device.type == 'cuda':
                addition = torch.rand(
                    outputs.size(0),
                    outputs.size(1),
                    outputs.size(2),
                    outputs.size(3),
                    offset_w).cuda()
            else:
                addition = torch.rand(
                    outputs.size(0),
                    outputs.size(1),
                    outputs.size(2),
                    outputs.size(3),
                    offset_w)
            outputs = torch.cat([outputs, addition], dim=4)

        out = torch.cat([inputs, outputs], dim=1)
        return out

def normalization(planes, norm='gn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(8, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m

class InitConv(nn.Module):
    def __init__(self, in_channels=1, out_channels=16, dropout=0.2):
        super(InitConv, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = dropout

    def forward(self, x):
        y = self.conv(x)
        y = F.dropout3d(y, self.dropout)
        
        return y

class EnBlock(nn.Module):
    def __init__(self, in_channels, norm='gn'):
        super(EnBlock, self).__init__()

        self.bn1 = normalization(in_channels, norm=norm)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

        self.bn2 = normalization(in_channels, norm=norm)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)
        y = self.bn2(x1)
        y = self.relu2(y)
        y = self.conv2(y)
        y = y + x

        return y

class EnDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EnDown, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        y = self.conv(x)

        return y

class DeUp_Cat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeUp_Cat, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, prev):
        x1 = self.conv1(x)
        y = self.conv2(x1)
        # y = y + prev
        y = torch.cat((prev, y), dim=1)
        y = self.conv3(y)
        return y

class DeBlock(nn.Module):
    def __init__(self, in_channels, norm='gn'):
        super(DeBlock, self).__init__()

        # self.bn1 = nn.BatchNorm3d(in_channels)
        self.bn1 = normalization(in_channels, norm=norm)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm3d(in_channels)
        self.bn2 = normalization(in_channels, norm=norm)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = x1 + x

        return x1

class Unet(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, num_classes=4):
        super(Unet, self).__init__()

        # self.InitConv = InitConv(in_channels=in_channels, out_channels=base_channels, dropout=0.2)
        self.InitConv = InitConv(in_channels=in_channels, out_channels=base_channels, dropout=0)

        self.EnBlock1 = EnBlock(in_channels=base_channels)
        self.EnDown1 = EnDown(in_channels=base_channels, out_channels=base_channels*2)

        self.EnBlock2_1 = EnBlock(in_channels=base_channels*2)
        self.EnBlock2_2 = EnBlock(in_channels=base_channels*2)
        self.EnDown2 = EnDown(in_channels=base_channels*2, out_channels=base_channels*4)

        self.EnBlock3_1 = EnBlock(in_channels=base_channels * 4)
        self.EnBlock3_2 = EnBlock(in_channels=base_channels * 4)
        self.EnDown3 = EnDown(in_channels=base_channels*4, out_channels=base_channels*8)

        self.EnBlock4_1 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_2 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_3 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_4 = EnBlock(in_channels=base_channels * 8)

        self.DeUpCat4 = DeUp_Cat(in_channels=base_channels * 8, out_channels=base_channels * 4)
        self.DeBlock4 = DeBlock(in_channels=base_channels*4)

        self.DeUpCat3 = DeUp_Cat(in_channels=base_channels * 4, out_channels=base_channels * 2)
        self.DeBlock3 = DeBlock(in_channels=base_channels*2)

        self.DeUpCat2 = DeUp_Cat(in_channels=base_channels * 2, out_channels=base_channels)
        self.DeBlock2 = DeBlock(in_channels=base_channels)
        self.final_conv = nn.Conv3d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.InitConv(x)       # (1, 16, 128, 128, 128)

        x1_1 = self.EnBlock1(x)
        x1_2 = self.EnDown1(x1_1)  # (1, 32, 64, 64, 64)

        x2_1 = self.EnBlock2_1(x1_2)
        x2_1 = self.EnBlock2_2(x2_1)
        x2_2 = self.EnDown2(x2_1)  # (1, 64, 32, 32, 32)

        x3_1 = self.EnBlock3_1(x2_2)
        x3_1 = self.EnBlock3_2(x3_1)
        x3_2 = self.EnDown3(x3_1)  # (1, 128, 16, 16, 16)

        x4_1 = self.EnBlock4_1(x3_2)
        x4_2 = self.EnBlock4_2(x4_1)
        x4_3 = self.EnBlock4_3(x4_2)
        x4_4 = self.EnBlock4_4(x4_3)  # (1, 128, 16, 16, 16)

        y4 = self.DeUpCat4(x4_4, x3_1)
        y4 = self.DeBlock4(y4) # (1, 64, 32, 32, 32)

        y3 = self.DeUpCat3(y4, x2_1)  # (1, 32, 64, 64, 64)
        y3 = self.DeBlock3(y3)

        y2 = self.DeUpCat2(y3, x1_1)  # (1, 16, 128, 128, 128)
        y2 = self.DeBlock2(y2)
        y = self.final_conv(y2)

        return y
