import math
from functools import reduce
from operator import __mul__
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from base_networks import DFFBlock, UBPBlock, DBPBlock
from arch import flows
import cygen
import time

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


class BasicEncoder(nn.Module):
    def __init__(self):
        super(BasicEncoder, self).__init__()

        self.conv_input_LR = nn.Conv2d(3, 16, 5, 1, 2)
        self.conv_input_HR = nn.Conv2d(3, 16, 5, 1, 2)

        self.down1_LR = DBPBlock(16)
        self.down2_LR = DBPBlock(32)

        self.down1_HR = DBPBlock(16)
        self.down2_HR = DBPBlock(32)

        self.fusion1 = nn.Conv2d(16, 16, 1, 1, 0)
        self.fusion2 = nn.Conv2d(32, 32, 1, 1, 0)
        self.fusion3 = nn.Conv2d(64, 64, 1, 1, 0)

    def forward(self, LR, HR):
        fm1x_LR = self.conv_input_LR(LR)
        fm2x_LR = self.down1_LR(fm1x_LR)
        fm4x_LR = self.down2_LR(fm2x_LR)

        fm1x_HR = self.conv_input_HR(HR) - self.fusion1(fm1x_LR)
        fm2x_HR = self.down1_HR(fm1x_HR) - self.fusion2(fm2x_LR)
        fm4x_HR = self.down2_HR(fm2x_HR) - self.fusion3(fm4x_LR)

        return fm4x_HR, [fm1x_LR, fm2x_LR, fm4x_LR]


class ResEncoder(nn.Module):
    def __init__(self, dim_z=128):
        super(ResEncoder, self).__init__()
        self.conv_output = nn.Sequential(
            ResidualBlock(64),
            nn.Conv2d(64, 32, 3, 2, 1),
            ResidualBlock(32)
        )

        self.pool_size = nn.Sequential(
            nn.MaxPool2d(4, 4),
            nn.PReLU()
        )
        # self.pool_channel = nn.Conv2d(64, 32, 1, 1, 0)

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.PReLU(),
            nn.Linear(1024, 512)
        )
        self.fc_mean = nn.Linear(512, dim_z)
        self.fc_var = nn.Sequential(
            nn.Linear(512, dim_z),
            nn.Softplus()
        )

    def forward(self, fm4x_HR):
        fm_output = self.conv_output(fm4x_HR)
        fm_pool_size = self.pool_size(fm_output)
        # fm_pool_channel = self.pool_channel(fm_pool_size)

        # fm_resize = fm_pool_channel.view(fm_pool_channel.shape[0], -1)
        fm_resize = fm_pool_size.view(fm_pool_size.shape[0], -1)
        h = self.fc(fm_resize)
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

        self.upConv0 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 6, 2, 2),
            ResidualBlock(32),
            nn.ConvTranspose2d(32, 64, 6, 2, 2),
            ResidualBlock(64),
            nn.ConvTranspose2d(64, 64, 6, 2, 2, bias=False)
        )

    def forward(self, z):
        fm_input = self.fc(z)
        fm_resize = fm_input.view(z.shape[0], 32, 8, 8)

        fm_res = self.upConv0(fm_resize)
        return fm_res


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.upConv1 = UBPBlock(64)
        self.upConv2 = UBPBlock(32)

        self.fusion1 = nn.Conv2d(16, 16, 1, 1, 0)
        self.fusion2 = nn.Conv2d(32, 32, 1, 1, 0)
        self.fusion3 = nn.Conv2d(64, 64, 1, 1, 0)

        self.conv_output = nn.Conv2d(16, 3, 5, 1, 2)

    def forward(self, fm_res, LR_fmList):
        fm4x = fm_res + self.fusion3(LR_fmList[2])
        fm4x = fm4x * 0.0
        fm2x = self.upConv1(fm4x) + self.fusion2(LR_fmList[1])
        fm1x = self.upConv2(fm2x) + self.fusion1(LR_fmList[0])

        out = self.conv_output(fm1x)
        return out


class SR_CyGen(nn.Module):
    def __init__(self, args):
        super(SR_CyGen, self).__init__()

        self.dim_z = args.dim_z
        self.batchSize = args.batch_size
        self.device = torch.device('cuda:' + str(args.gpu) if args.gpu >= 0 and args.cuda else 'cpu')

        self.basicEncoder = BasicEncoder()
        self.resEncoder = ResEncoder(dim_z=self.dim_z)
        self.flowEncoder = FlowEncoder(dim_z=self.dim_z)
        self.resDecoder = ResDecoder(dim_z=self.dim_z)
        self.decoder = Decoder()

        self.LR = None
        self.LR_fmList = None
        self.fm_res_out = None
        self.w_cm = args.w_cm
        self.w_px = args.w_px

        self.x_gen_stepsize = 1e-3
        self.z_gen_stepsize = 1e-4
        self.x_gen_anneal = 10.
        self.z_gen_anneal = 10.
        self.x_gen_freeze = None
        self.z_gen_freeze = None

        self.frame = cygen.CyGen_FlowqNoInv(args.cmtype, args.pxtype,
                self.eval_logp,
                self.draw_q0,
                lambda x, eps: self.eval_z1eps_logqt(x, eps, eval_jac=True),
                self.eval_z1eps,
                args.w_cm, args.w_px,
                args.n_mc_cm, args.n_mc_px)

        # self.start_time = 0.0

    def draw_q0(self, batchSize):
        eps = torch.randn((batchSize, self.dim_z), device=self.device)
        eps.data.clamp_(-100.0, 100.0)
        return eps

    def setLR(self, LR):
        self.LR = LR

    def eval_z1eps_logqt(self, fm_res_in, eps, eval_jac=False):
        # fm_res_in, self.LR_fmList = self.basicEncoder(self.LR, SR)
        h, z_mean, z_var = self.resEncoder(fm_res_in)
        z, logp, jaceps_z = self.flowEncoder.eval_z1eps_logqt(h, z_mean, z_var, eps, eval_jac)
        return z, logp, jaceps_z

    def eval_z1eps(self, fm_res_in, eps):
        # self.fm_res_in, self.LR_fmList = self.basicEncoder(self.LR, SR)
        h, z_mean, z_var = self.resEncoder(fm_res_in)
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

    def getlosses(self, LR, SR):
        # self.start_time = time.time()

        self.setLR(LR)
        fm_res_in, self.LR_fmList = self.basicEncoder(self.LR, SR)

        cm_loss = self.frame.get_cmloss(fm_res_in)
        MSE_loss_2 = F.mse_loss(fm_res_in, self.fm_res_out)
        # with torch.no_grad():
        #     fm_avg = torch.mean(torch.pow(fm_res_in, 2))
        fm_avg = torch.mean(torch.pow(fm_res_in, 2))
        MSE_loss_2 = MSE_loss_2 / fm_avg
        print(fm_avg.item(), MSE_loss_2.item())

        SR_predict = self.decoder(self.fm_res_out, self.LR_fmList)

        MSE_loss = F.mse_loss(SR_predict, SR)
        VGG_loss = torch.tensor(0.0, device=self.device)
        loss = self.w_cm * cm_loss + self.w_px * MSE_loss + 0.2 * MSE_loss_2
        return [loss, cm_loss, MSE_loss, VGG_loss]

    def getlosses_noCyGen(self, LR, SR):
        self.setLR(LR)
        eps = self.draw_q0(LR.shape[0])
        z = self.eval_z1eps(SR, eps)
        x = self.draw_p(z)
        MSE_loss = F.mse_loss(self.SR_predict, SR)
        cm_loss = torch.tensor(0.0, device=self.device)
        VGG_loss = torch.tensor(0.0, device=self.device)
        loss = self.w_cm * cm_loss + self.w_px * MSE_loss
        return [loss, cm_loss, MSE_loss, VGG_loss]

    def generate(self, gentype, LR, n_iter1=6, n_iter2=10):
        result = []
        self.setLR(LR)
        zp = self.draw_q0(LR.shape[0])
        fm_res_in, self.LR_fmList = self.basicEncoder(LR, LR)
        for i in range(n_iter1):
            if gentype == "gibbs":
                model_samples_ori_z = self.frame.generate("gibbs", self.draw_p, n_iter2, z0=zp)
            elif gentype == "langv-x":
                model_samples_ori_z = self.frame.generate("langv-x", self.draw_p, n_iter2, z0=zp,
                                                          stepsize=self.x_gen_stepsize,
                                                          anneal=self.x_gen_anneal, freeze=self.x_gen_freeze,
                                                          x_range=[0., 1.])
            elif gentype == "langv-z":
                model_samples_ori_z = self.frame.generate("langv-z", self.draw_p, n_iter2, z0=zp,
                                                          stepsize=self.z_gen_stepsize,
                                                          anneal=self.z_gen_anneal, freeze=self.z_gen_freeze)
            fm_res_out = model_samples_ori_z[0]
            SR_pred = self.decoder(fm_res_out, self.LR_fmList)
            fm_res_in, self.LR_fmList = self.basicEncoder(self.LR, SR_pred)

            result.append(SR_pred)

        return result

    def forward(self, LR, SR):
        self.setLR(LR)
        eps = self.draw_q0(LR.shape[0])
        z = self.eval_z1eps(SR, eps)
        x = self.draw_p(z)
        return x


