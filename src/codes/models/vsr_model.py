import torch
import torch.optim as optim
from collections import OrderedDict
from .base_model import BaseModel
from .networks import define_generator
from .networks.vgg_nets import VGGFeatureExtractor
from .optim import define_criterion, define_lr_schedule
from utils import net_utils, data_utils


class VSRModel(BaseModel):
    """ A model wraper for objective video super-resolution

        It contains a generator, as well as relative functions to train and test
        the generator
    """

    def __init__(self, opt):
        super(VSRModel, self).__init__(opt)

        if self.verbose:
            self.logger.info('{} Model Info {}'.format('=' * 20, '=' * 20))
            self.logger.info('Model: {}'.format(opt['model']['name']))

        # set network
        self.set_network()

        # configs for training
        if self.is_train:
            self.config_training()

    def set_network(self):
        # define generator net
        self.net_G = define_generator(self.opt).to(self.device)
        if self.verbose:
            self.logger.info('Generator: {}\n'.format(
                self.opt['model']['generator']['name']) + self.net_G.__str__())

        # load network
        load_path_G = self.opt['model']['generator'].get('load_path')
        load_path_reconstr = self.opt['model']['generator'].get('reconstruction_path')
        if load_path_G is not None:
            self.load_network(self.net_G, load_path_G)
            if self.verbose:
                self.logger.info('Loaded generator from: {}'.format(load_path_G))
        elif load_path_reconstr is not None:
            # Load the reconstruction block
            self.load_reconstruction_block(self.net_G, load_path_reconstr)
            if self.verbose:
                self.logger.info('Loaded reconstruction module from: {}'.format(load_path_reconstr))


    def config_training(self):
        # set criterion
        self.set_criterion()

        # set optimizer
        self.optim_G = optim.Adam(
            self.net_G.parameters(),
            lr=self.opt['train']['generator']['lr'],
            weight_decay=self.opt['train']['generator'].get('weight_decay', 0),
            betas=(
                self.opt['train']['generator'].get('beta1', 0.9),
                self.opt['train']['generator'].get('beta2', 0.999)))

        # set lr schedule
        self.sched_G = define_lr_schedule(
            self.opt['train']['generator'].get('lr_schedule'), self.optim_G)

    def set_criterion(self):
        # pixel criterion
        self.pix_crit = define_criterion(
            self.opt['train'].get('pixel_crit'))

        # warping criterion
        self.warp_crit = define_criterion(
            self.opt['train'].get('warping_crit'))
        
        # alignment crit -> DCN alignment
        self.align_crit = define_criterion(
            self.opt['train'].get('align_crit'))

        # feature criterion
        self.feat_crit = define_criterion(
            self.opt['train'].get('feature_crit'))
        if self.feat_crit is not None:  # load feature extractor
            feature_layers = self.opt['train']['feature_crit'].get(
                'feature_layers', ['conv1_2', 'conv2_2', 'conv3_4', 'conv4_4', 'conv5_4'])
            self.feature_weights = self.opt['train']['feature_crit'].get(
                'feature_weights', [0.1, 0.1, 1, 1, 1])
            self.net_F = VGGFeatureExtractor(feature_layers).to(self.device)

        # flow & mask criterion
        self.flow_crit = define_criterion(
            self.opt['train'].get('flow_crit'))

        # ping-pong criterion
        self.pp_crit = define_criterion(
            self.opt['train'].get('pingpong_crit'))

    def train(self, data):
        """ Function of mini-batch training

            Parameters:
                :param data: a batch of training data (lr & gt) in shape ntchw
        """

        # ------------ prepare data ------------ #
        lr_data, gt_data = data['lr'], data['gt']


        # ------------ clear optim ------------ #
        self.net_G.train()
        self.optim_G.zero_grad()


        # ------------ forward G ------------ #
        net_G_output_dict = self.net_G.forward_sequence(lr_data)
        hr_data = net_G_output_dict['hr_data']


        # ------------ optimize G ------------ #
        loss_G = 0
        self.log_dict = OrderedDict()

        # pixel loss
        pix_w = self.opt['train']['pixel_crit'].get('weight', 1)
        loss_pix_G = pix_w * self.pix_crit(hr_data, gt_data)
        loss_G += loss_pix_G
        self.log_dict['l_pix_G'] = loss_pix_G.item()

        # warping loss -> motion estimation and compensation
        if self.warp_crit is not None: 
            # warp lr_prev according to lr_flow
            lr_curr = net_G_output_dict['lr_curr']
            lr_prev = net_G_output_dict['lr_prev']
            lr_flow = net_G_output_dict['lr_flow']
            lr_warp = net_utils.backward_warp(lr_prev, lr_flow)

            warp_w = self.opt['train']['warping_crit'].get('weight', 1)
            loss_warp_G = warp_w * self.warp_crit(lr_warp, lr_curr)
            loss_G += loss_warp_G
            self.log_dict['l_warp_G'] = loss_warp_G.item()

        # alignment crit -> feature alignment
        if self.align_crit is not None:
            # Get tensors
            win_size = self.net_G.num_frames

            # Repeat the current frame on dim 2 to allow the loss computation
            lr_curr = lr_data.unsqueeze(2).repeat(1, 1, win_size - 1, 1, 1, 1)  # B x T x Num_frames-1 x C x H x W
            lr_aligned = net_G_output_dict['lr_aligned_feat']  # B x T x Num_frames-1 x C x H x W

            align_w = self.opt['train']['align_crit'].get('weight', 1)            
            loss_align_G = align_w * self.align_crit(lr_curr, lr_aligned)
            loss_G += loss_align_G
            self.log_dict['l_align_G'] = loss_align_G.item()

        # feature (feat) loss
        if self.feat_crit is not None:
            _, _, c, gt_h, gt_w = gt_data.size()
            hr_merge = hr_data.view(-1, c, gt_h, gt_w)
            gt_merge = gt_data.view(-1, c, gt_h, gt_w)

            hr_feat_lst = self.net_F(hr_merge)
            gt_feat_lst = self.net_F(gt_merge)
            loss_feat_G = 0
            i = 0
            for hr_feat, gt_feat in zip(hr_feat_lst, gt_feat_lst):
                loss_feat_G += self.feat_crit(hr_feat, gt_feat.detach()) * self.feature_weights[i]
                i += 1

            feat_w = self.opt['train']['feature_crit'].get('weight', 1)
            loss_feat_G = feat_w * loss_feat_G
            loss_G += loss_feat_G
            self.log_dict['l_feat_G'] = loss_feat_G.item()

        # ping-pong (pp) loss
        if self.pp_crit is not None:
            tempo_extent = self.opt['train']['tempo_extent']
            hr_data_fw = hr_data[:, :tempo_extent - 1, ...]     # -------->|
            hr_data_bw = hr_data[:, 1:, ...].flip(1) # <--------|

            pp_w = self.opt['train']['pingpong_crit'].get('weight', 1)
            loss_pp_G = pp_w * self.pp_crit(hr_data_fw, hr_data_bw)
            loss_G += loss_pp_G
            self.log_dict['l_pp_G'] = loss_pp_G.item()

        # optimize
        loss_G.backward()
        self.optim_G.step()

    def infer(self, lr_data):
        """ Function of inference

            Parameters:
                :param lr_data: a rgb video sequence with shape thwc
                :return: a rgb video sequence with type np.uint8 and shape thwc
        """
        # Eval mode
        self.net_G.eval()

        # canonicalize
        lr_data = data_utils.canonicalize(lr_data)  # to torch.FloatTensor

        # temporal padding
        lr_data, n_pad_front = self.pad_sequence(lr_data)

        # infer
        hr_seq = self.net_G.infer_sequence(lr_data, self.device)
        hr_seq = hr_seq[n_pad_front:, ...]

        # Train mode
        self.net_G.train()

        return hr_seq

    def save(self, current_iter):
        self.save_network(self.net_G, 'G', current_iter)
