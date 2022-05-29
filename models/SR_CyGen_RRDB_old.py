import math
from functools import reduce, partial
from operator import __mul__
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.base_networks import UBPBlock, DBPBlock, make_layer, RRDB
from arch import flows
import models.cygen as cygen


# 将各元素变成常量
def tensorify(device=None, *args) -> tuple:
    return tuple(arg.to(device) if isinstance(arg, torch.Tensor) else torch.tensor(arg, device=device) for arg in args)


def eval_logp_normal(x, mean = 0., var = 1., ndim = 1):
    mean = tensorify(x.device, mean)[0]
    var = tensorify(x.device, var)[0].expand(x.shape)
    if ndim == 0:
        x = x.unsqueeze(-1); mean = mean.unsqueeze(-1); var = var.unsqueeze(-1)
        ndim = 1
    reduce_dims = tuple(range(-1, -ndim-1, -1))
    quads = ((x-mean)**2 / var).sum(dim=reduce_dims)
    log_det = var.log().sum(dim=reduce_dims)
    numel = reduce(__mul__, x.shape[-ndim:])
    return -.5 * (quads + log_det + numel * math.log(2*math.pi))


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out


class LREncoder(nn.Module):
    def __init__(self, n_RRDB=8):
        super(LREncoder, self).__init__()
        RRDB_block_f = partial(RRDB, nf=32, gc=32)
        self.conv_input = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1)
        )
        self.RRDB_trunk = make_layer(RRDB_block_f, n_RRDB)
        self.trunk_conv = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        fm = self.conv_input(x)
        fm = self.conv1(fm)
        fm = self.conv2(fm)
        trunk = self.trunk_conv(self.RRDB_trunk(fm))
        fm = fm + trunk
        return fm


class HREncoder(nn.Module):
    def __init__(self, n_RRDB=8):
        super(HREncoder, self).__init__()
        RRDB_block_f = partial(RRDB, nf=32, gc=32)

        self.conv_input = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1)
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1)
        )
        self.RRDB_trunk = make_layer(RRDB_block_f, n_RRDB)
        self.trunk_conv = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        fm_4x = self.conv_input(x)
        fm_2x = self.down1(fm_4x)
        fm_1x = self.down2(fm_2x)

        trunk = self.trunk_conv(self.RRDB_trunk(fm_1x))
        fm = fm_1x + trunk
        return fm


class ResEncoder(nn.Module):
    def __init__(self, dim_z=128):
        super(ResEncoder, self).__init__()
        self.conv_res = nn.Sequential(
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.Conv2d(16, 16, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(16, 8, 3, 1, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.PReLU()
        )
        self.fc_mean = nn.Linear(512, dim_z)
        self.fc_var = nn.Sequential(
            nn.Linear(512, dim_z),
            nn.Softplus()
        )

    def forward(self, x):
        fm_res = self.conv_res(x)
        fm_vec = fm_res.view(fm_res.shape[0], -1)
        h = self.fc(fm_vec)
        z_mean = self.fc_mean(h)
        z_var = self.fc_var(h)

        return h, z_mean, z_var


class FlowEncoder(nn.Module):
    def __init__(self, dim_z=128):
        super(FlowEncoder, self).__init__()

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.

        self.num_flows = 4
        self.num_householder = 4
        self.dim_z = dim_z
        self.q_nn_output_dim = 512
        flow = flows.Sylvester

        identity = torch.eye(self.dim_z, self.dim_z)
        # Add batch dimension
        identity = identity.unsqueeze(0)
        self.register_buffer('_eye', Variable(identity))
        self._eye.requires_grad = False

        # Masks needed for triangular r1 and r2.
        triu_mask = torch.triu(torch.ones(self.dim_z, self.dim_z), diagonal=1)
        triu_mask = triu_mask.unsqueeze(0).unsqueeze(3)
        diag_idx = torch.arange(0, self.dim_z).long()

        self.register_buffer('triu_mask', Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer('diag_idx', diag_idx)

        # Amortized flow parameters
        # Diagonal elements of r1 * r2 have to satisfy -1 < r1 * r2 for flow to be invertible
        self.diag_activation = nn.Tanh()

        self.amor_d = nn.Linear(self.q_nn_output_dim, self.num_flows * self.dim_z * self.dim_z)

        self.amor_diag1 = nn.Sequential(
            nn.Linear(self.q_nn_output_dim, self.num_flows * self.dim_z),
            self.diag_activation
        )
        self.amor_diag2 = nn.Sequential(
            nn.Linear(self.q_nn_output_dim, self.num_flows * self.dim_z),
            self.diag_activation
        )

        self.amor_q = nn.Linear(self.q_nn_output_dim, self.num_flows * self.dim_z * self.num_householder)

        self.amor_b = nn.Linear(self.q_nn_output_dim, self.num_flows * self.dim_z)

        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow(self.dim_z)

            self.add_module('flow_' + str(k), flow_k)

    def batch_construct_orthogonal(self, q):
        """
        Batch orthogonal matrix construction.
        :param q:  q contains batches of matrices, shape : (batch_size, num_flows * dim_z * num_householder)
        :return: batches of orthogonalized matrices, shape: (batch_size * num_flows, dim_z, dim_z)
        """

        # Reshape to shape (num_flows * batch_size * num_householder, dim_z)
        batch_size = q.shape[:-1]
        q = q.view(-1, self.dim_z)

        norm = torch.norm(q, p=2, dim=1, keepdim=True)  # ||v||_2
        v = torch.div(q, norm)  # v / ||v||_2

        # Calculate Householder Matrices
        vvT = torch.bmm(v.unsqueeze(2), v.unsqueeze(1))  # v * v_T : batch_dot( B x L x 1 * B x 1 x L ) = B x L x L

        amat = self._eye - 2 * vvT  # NOTICE: v is already normalized! so there is no need to calculate vvT/vTv

        ## Reshaping: first dimension is batch_size * num_flows
        # amat = amat.view(-1, self.num_householder, self.dim_z, self.dim_z)

        # tmp = amat[:, 0]
        # for k in range(1, self.num_householder):
        #    tmp = torch.bmm(amat[:, k], tmp)

        # amat = tmp.view(*batch_size, self.num_flows, self.dim_z, self.dim_z)
        ##amat = amat.transpose(0, 1)
        # ndim_batch = len(batch_size)
        # amat = amat.permute([ndim_batch] + list(range(ndim_batch)) + [-2,-1])

        # return amat

        amat = amat.view(self.num_householder, self.num_flows, *batch_size, self.dim_z, self.dim_z)
        return reduce(torch.matmul, amat.unbind(0))

    def encoder(self, h):
        # Amortized r1, r2, q, b for all flows
        batch_size = h.shape[0]
        full_d = self.amor_d(h)
        diag1 = self.amor_diag1(h)
        diag2 = self.amor_diag2(h)

        full_d = full_d.view(batch_size, self.dim_z, self.dim_z, self.num_flows)
        diag1 = diag1.view(batch_size, self.dim_z, self.num_flows)
        diag2 = diag2.view(batch_size, self.dim_z, self.num_flows)

        r1 = full_d * self.triu_mask
        r2 = full_d.transpose(-3, -2) * self.triu_mask

        r1[..., self.diag_idx, self.diag_idx, :] = diag1
        r2[..., self.diag_idx, self.diag_idx, :] = diag2

        q = self.amor_q(h)

        b = self.amor_b(h)

        # Resize flow parameters to divide over K flows
        b = b.view(batch_size, 1, self.dim_z, self.num_flows)
        return r1, r2, q, b

    def eval_z1eps_logqt(self, h, z_mu, z_var, eps, eval_jac=False):
        r1, r2, q, b = self.encoder(h)
        q_ortho = self.batch_construct_orthogonal(q)

        z_std = z_var.sqrt()
        self.log_det_j = z_std.log().sum(dim=-1)
        jaceps_z = z_std.diag_embed() if eval_jac else None

        z = [z_mu + z_std * eps]
        # Normalizing flows
        for k in range(self.num_flows):
            flow_k = getattr(self, 'flow_' + str(k))
            q_k = q_ortho[k]
            # print(z[k].size(), r1[..., k].size(), q_k.size(), b[..., k].size())
            z_k, log_det_jacobian, jaczk_zkp1 = flow_k(z[k], r1[..., k], r2[..., k], q_k, b[..., k],
                                                       sum_ldj=True, eval_jac=eval_jac)
            z.append(z_k)
            self.log_det_j = self.log_det_j + log_det_jacobian
            if eval_jac: jaceps_z = jaceps_z @ jaczk_zkp1
        # Evaluate log-density
        logpeps = eval_logp_normal(eps, 0., 1., ndim=1)
        logp = logpeps - self.log_det_j
        return z[-1], logp, jaceps_z

    # encoder
    def eval_z1eps(self, h, z_mu, z_var, eps):
        r1, r2, q, b = self.encoder(h)
        z_std = z_var.sqrt()
        # Orthogonalize all q matrices
        q_ortho = self.batch_construct_orthogonal(q)
        # Sample z_0
        z = [z_mu + z_std * eps]
        # Normalizing flows
        for k in range(self.num_flows):
            flow_k = getattr(self, 'flow_' + str(k))
            q_k = q_ortho[k]
            z_k = flow_k.draw(z[k], r1[..., k], r2[..., k], q_k, b[..., k])
            z.append(z_k)
        return z[-1]

    def forward(self, h, z_mu, z_var, eps):
        r1, r2, q, b = self.encoder(h)

        q_ortho = self.batch_construct_orthogonal(q)

        # Normalizing flows
        z_std = z_var.sqrt()
        z = [z_mu + z_std * eps]
        for k in range(self.num_flows):
            flow_k = getattr(self, 'flow_' + str(k))
            q_k = q_ortho[k]
            z_k = flow_k.draw(z[k], r1[..., k], r2[..., k], q_k, b[..., k])
            z.append(z_k)
        return z[-1]


class ResDecoder(nn.Module):
    def __init__(self, dim_z=128):
        super(ResDecoder, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(dim_z, 512),
            nn.PReLU(),
            nn.Linear(512, 2048)
        )

        self.conv_res = nn.Sequential(
            nn.ConvTranspose2d(8, 8, 6, 2, 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.ConvTranspose2d(16, 16, 6, 2, 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(16, 32, 3, 1, 1)
        )

    def forward(self, z):
        fm_vec = self.fc(z)
        fm_res = fm_vec.view(z.shape[0], 8, 16, 16)

        fm_res = self.conv_res(fm_res)
        return fm_res


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.up1 = nn.Sequential(
            UBPBlock(32),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.up2 = nn.Sequential(
            UBPBlock(32),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv_output = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 3, 3, 1, 1)
        )

    def forward(self, x):
        fm_2x = self.up1(x)
        fm_4x = self.up2(fm_2x)

        out = self.conv_output(fm_4x)
        return out


class SR_CyGen(nn.Module):
    def __init__(self, opt):
        super(SR_CyGen, self).__init__()

        self.dim_z = opt['network']['dim_z']
        self.n_RRDB = opt['network']['n_RRDB']
        self.batchSize = opt['batch_size']
        self.device = opt['device']

        self.encoder_HR = HREncoder(n_RRDB=self.n_RRDB)
        self.encoder_LR = LREncoder(n_RRDB=self.n_RRDB)
        self.resEncoder = ResEncoder(dim_z=self.dim_z)
        self.flowEncoder = FlowEncoder(dim_z=self.dim_z)
        self.resDecoder = ResDecoder(dim_z=self.dim_z)
        self.decoder = Decoder()

        self.LR = None
        self.fm_LR = None
        self.fm_res_out = None
        self.w_mse = opt['network']['w_mse']
        self.w_cm_1 = opt['network']['w_cm_1']
        self.w_cm_2 = opt['network']['w_cm_2']

        self.x_gen_stepsize = 1e-3
        self.z_gen_stepsize = 1e-4
        self.x_gen_anneal = 10.
        self.z_gen_anneal = 10.
        self.x_gen_freeze = None
        self.z_gen_freeze = None

        self.frame = cygen.CyGen_FlowqNoInv(opt['network']['cmtype'], opt['network']['pxtype'],
                self.eval_logp,
                self.draw_q0,
                lambda x, eps: self.eval_z1eps_logqt(x, eps, eval_jac=True),
                self.eval_z1eps,
                )

    def draw_q0(self, batch_size):
        eps = torch.randn((batch_size, self.dim_z), device=self.device)
        eps.data.clamp_(-100.0, 100.0)
        return eps

    def generate_noise(self, x):
        with torch.no_grad():
            std = torch.std(x)
        noise = torch.randn(x.shape) * std
        return noise

    def setLR(self, LR):
        self.LR = LR
        self.fm_LR = self.encoder_LR(self.LR)

    def eval_z1eps_logqt(self, fm_res, eps, eval_jac=False):
        h, z_mean, z_var = self.resEncoder(fm_res)
        z, logp, jaceps_z = self.flowEncoder.eval_z1eps_logqt(h, z_mean, z_var, eps, eval_jac)
        return z, logp, jaceps_z

    def eval_z1eps(self, fm_res, eps):
        h, z_mean, z_var = self.resEncoder(fm_res)
        return self.flowEncoder.eval_z1eps(h, z_mean, z_var, eps)

    def eval_logp(self, fm_res_in, z):
        self.fm_res_out = self.resDecoder(z)
        return self.eval_logp_normal(fm_res_in, self.fm_res_out)

    def eval_logp_normal(self, x, mean=0., var=1., ndim=1):
        mean = tensorify(x.device, mean)[0]
        var = tensorify(x.device, var)[0].expand(x.shape)
        if ndim == 0:
            x = x.unsqueeze(-1);
            mean = mean.unsqueeze(-1);
            var = var.unsqueeze(-1)
            ndim = 1
        reduce_dims = tuple(range(-1, -ndim - 1, -1))
        quads = ((x - mean) ** 2 / var).sum(dim=reduce_dims)
        log_det = var.log().sum(dim=reduce_dims)
        numel = reduce(__mul__, x.shape[-ndim:])
        result =  -.5 * (quads + log_det + numel * math.log(2 * math.pi))

        # print("Forward Time1:" + str(time.time() - self.start_time))
        # self.start_time = time.time()
        return result

    def draw_p(self, z, num=1):
        self.fm_res_out = self.resDecoder(z)
        return self.fm_res_out

    def getlosses(self, LR, HR):
        self.setLR(LR)
        fm_HR = self.encoder_HR(HR)
        fm_res_in = fm_HR - self.fm_LR

        cm_loss_1 = self.frame.get_cmloss(fm_res_in)
        cm_loss_2 = F.mse_loss(fm_res_in, self.fm_res_out)
        # with torch.no_grad():
        #     fm_avg = torch.mean(torch.pow(fm_res_in, 2))
        fm_avg = torch.mean(torch.pow(fm_res_in, 2))
        cm_loss_2 = cm_loss_2 / fm_avg
        # print(fm_avg.item(), cm_loss_2.item())

        fm_SR = self.fm_res_out + self.fm_LR
        SR_predict = self.decoder(fm_SR)

        MSE_loss = F.mse_loss(SR_predict, HR)
        loss = self.w_mse * MSE_loss + self.w_cm_1 * cm_loss_1 + self.w_cm_2 * cm_loss_2
        # print("{:.4f} | {:.4f} | {:.6f} | {:.6f}".format(loss.item(), MSE_loss.item(), cm_loss_1.item(), cm_loss_2.item()))
        return [loss, MSE_loss, cm_loss_1, cm_loss_2]

    def get_loss_noCyGen(self, LR, HR):
        SR_pred = self.forward(LR, HR)
        loss = F.mse_loss(SR_pred, HR)
        return [loss, loss, torch.tensor(0.0), torch.tensor(0.0)]

    def generate(self, LR, n_iter1=6, n_iter2=10):
        result = []
        self.setLR(LR)
        LR_resize = F.interpolate(LR, scale_factor=4.0, mode='bicubic')
        fm_HR = self.encoder_HR(LR_resize)
        fm_res_in = fm_HR - self.fm_LR
        zp = self.draw_q0(LR.shape[0])
        xp = fm_res_in

        for i in range(n_iter1):
            model_samples_ori_z = self.frame.generate("gibbs", self.draw_p, n_iter2, x0=xp)

            fm_res_out = model_samples_ori_z[0]
            fm_SR = fm_res_out + self.fm_LR
            SR_pred = self.decoder(fm_SR)
            result.append(SR_pred)

            fm_HR = self.encoder_HR(SR_pred)
            fm_res_in = fm_HR - self.fm_LR
            xp = fm_res_in

        return result

    def forward(self, LR, HR):
        self.setLR(LR)
        eps = self.draw_q0(LR.shape[0])

        fm_HR = self.encoder_HR(HR)
        fm_res_in = fm_HR + self.fm_LR
        z = self.eval_z1eps(fm_res_in, eps)
        fm_res_out = self.draw_p(z)
        fm_SR = fm_res_out + self.fm_LR
        SR = self.decoder(fm_SR)
        return SR


