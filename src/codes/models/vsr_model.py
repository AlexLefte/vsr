from collections import OrderedDict

import torch
import torch.optim as optim
from .base_model import BaseModel
from .networks import define_generator
from .networks.vgg_nets import VGGFeatureExtractor
from .optim import define_criterion, define_lr_schedule
from utils import net_utils, data_utils


# class VSRRecurrentModel(BaseModel):
#     """ A model wraper for objective video super-resolution

#         It contains a generator, as well as relative functions to train and test
#         the generator
#     """

#     def __init__(self, opt):
#         super(VSRRecurrentModel, self).__init__(opt)

#         if self.verbose:
#             self.logger.info('{} Model Info {}'.format('=' * 20, '=' * 20))
#             self.logger.info('Model: {}'.format(opt['model']['name']))

#         # set network
#         self.set_network()

#         # configs for training
#         if self.is_train:
#             self.config_training()

#     def set_network(self):
#         # define net G
#         self.net_G = define_generator(self.opt).to(self.device)
#         if self.verbose:
#             self.logger.info('Generator: {}\n'.format(
#                 self.opt['model']['generator']['name']) + self.net_G.__str__())

#         # load network
#         load_path_G = self.opt['model']['generator'].get('load_path')
#         if load_path_G is not None:
#             self.load_network(self.net_G, load_path_G)
#             if self.verbose:
#                 self.logger.info('Load generator from: {}'.format(load_path_G))

#     def config_training(self):
#         # set criterion
#         self.set_criterion()

#         # set optimizer
#         self.optim_G = optim.Adam(
#             self.net_G.parameters(),
#             lr=self.opt['train']['generator']['lr'],
#             weight_decay=self.opt['train']['generator'].get('weight_decay', 0),
#             betas=(
#                 self.opt['train']['generator'].get('beta1', 0.9),
#                 self.opt['train']['generator'].get('beta2', 0.999)))

#         # set lr schedule
#         self.sched_G = define_lr_schedule(
#             self.opt['train']['generator'].get('lr_schedule'), self.optim_G)

#     def set_criterion(self):
#         # pixel criterion
#         self.pix_crit = define_criterion(
#             self.opt['train'].get('pixel_crit'))

#         # warping criterion
#         self.warp_crit = define_criterion(
#             self.opt['train'].get('warping_crit'))
        
#         # feature criterion
#         self.feat_crit = define_criterion(
#             self.opt['train'].get('feature_crit'))
#         if self.feat_crit is not None:  # load feature extractor
#             feature_layers = self.opt['train']['feature_crit'].get(
#                 'feature_layers', [8, 17, 26, 35])
#             self.net_F = VGGFeatureExtractor(feature_layers).to(self.device)

#         # flow & mask criterion
#         self.flow_crit = define_criterion(
#             self.opt['train'].get('flow_crit'))

#         # ping-pong criterion
#         self.pp_crit = define_criterion(
#             self.opt['train'].get('pingpong_crit'))

#     def train(self, data):
#         """ Function of mini-batch training

#             Parameters:
#                 :param data: a batch of training data (lr & gt) in shape ntchw
#         """

#         # ------------ prepare data ------------ #
#         lr_data, gt_data = data['lr'], data['gt']


#         # ------------ clear optim ------------ #
#         self.net_G.train()
#         self.optim_G.zero_grad()


#         # ------------ forward G ------------ #
#         net_G_output_dict = self.net_G.forward_sequence(lr_data)
#         hr_data = net_G_output_dict['hr_data']


#         # ------------ optimize G ------------ #
#         loss_G = 0
#         self.log_dict = OrderedDict()

#         # pixel loss
#         pix_w = self.opt['train']['pixel_crit'].get('weight', 1)
#         loss_pix_G = pix_w * self.pix_crit(hr_data, gt_data)
#         loss_G += loss_pix_G
#         self.log_dict['l_pix_G'] = loss_pix_G.item()

#         # warping loss
#         if self.warp_crit is not None:
#             # warp lr_prev according to lr_flow
#             lr_curr = net_G_output_dict['lr_curr']
#             lr_prev = net_G_output_dict['lr_prev']
#             lr_flow = net_G_output_dict['lr_flow']
#             lr_warp = net_utils.backward_warp(lr_prev, lr_flow)

#             warp_w = self.opt['train']['warping_crit'].get('weight', 1)
#             loss_warp_G = warp_w * self.warp_crit(lr_warp, lr_curr)
#             loss_G += loss_warp_G
#             self.log_dict['l_warp_G'] = loss_warp_G.item()

#         # feature (feat) loss
#         if self.feat_crit is not None:
#             _, _, c, gt_h, gt_w = gt_data.size()
#             hr_merge = hr_data.view(-1, c, gt_h, gt_w)
#             gt_merge = gt_data.view(-1, c, gt_h, gt_w)

#             hr_feat_lst = self.net_F(hr_merge)
#             gt_feat_lst = self.net_F(gt_merge)
#             loss_feat_G = 0
#             for hr_feat, gt_feat in zip(hr_feat_lst, gt_feat_lst):
#                 loss_feat_G += self.feat_crit(hr_feat, gt_feat.detach())

#             feat_w = self.opt['train']['feature_crit'].get('weight', 1)
#             loss_feat_G = feat_w * loss_feat_G
#             loss_G += loss_feat_G
#             self.log_dict['l_feat_G'] = loss_feat_G.item()

#         # ping-pong (pp) loss
#         if self.pp_crit is not None:
#             tempo_extent = self.opt['train']['tempo_extent']
#             hr_data_fw = hr_data[:, :tempo_extent - 1, ...]     # -------->|
#             # Alex L: took the frames starting with the second and mirrored afterwards
#             hr_data_bw = hr_data[:, 1:, ...].flip(1) # <--------|

#             pp_w = self.opt['train']['pingpong_crit'].get('weight', 1)
#             loss_pp_G = pp_w * self.pp_crit(hr_data_fw, hr_data_bw)
#             loss_G += loss_pp_G
#             self.log_dict['l_pp_G'] = loss_pp_G.item()

#         # optimize
#         loss_G.backward()
#         self.optim_G.step()

#     def infer(self, lr_data):
#         """ Function of inference

#             Parameters:
#                 :param lr_data: a rgb video sequence with shape thwc
#                 :return: a rgb video sequence with type np.uint8 and shape thwc
#         """
#         print(lr_data.size())

#         # Eval mode
#         self.net_G.eval()

#         # canonicalize
#         lr_data = data_utils.canonicalize(lr_data)  # to torch.FloatTensor
#         # lr_data = lr_data.permute(0, 3, 1, 2)  # tchw

#         # temporal padding
#         lr_data, n_pad_front = self.pad_sequence(lr_data)

#         # infer
#         hr_seq = self.net_G.infer_sequence(lr_data, self.device)
#         hr_seq = hr_seq[n_pad_front:, ...]

#         # Train mode
#         self.net_G.train()

#         return hr_seq

#     def save(self, current_iter):
#         self.save_network(self.net_G, 'G', current_iter)


# class VSRPatchModel(BaseModel):
#     """ A model wrapper for objective video super-resolution using sliding window approach

#         It contains a generator that processes video frames in a sliding window fashion
#         rather than using recurrent connections.
#     """

#     def __init__(self, opt):
#         super(VSRPatchModel, self).__init__(opt)

#         if self.verbose:
#             self.logger.info('{} Model Info {}'.format('=' * 20, '=' * 20))
#             self.logger.info('Model: {}'.format(opt['model']['name']))

#         # Set sliding window parameters
#         self.window_size = opt['model'].get('window_size', 5)  # Default window size of 5 frames
#         self.window_stride = opt['model'].get('window_stride', 1)  # Default stride of 1 frame
        
#         # set network
#         self.set_network()

#         # configs for training
#         if self.is_train:
#             self.config_training()

#     def set_network(self):
#         # define net G
#         self.net_G = define_generator(self.opt).to(self.device)
#         if self.verbose:
#             self.logger.info('Generator: {}\n'.format(
#                 self.opt['model']['generator']['name']) + self.net_G.__str__())

#         # load network
#         load_path_G = self.opt['model']['generator'].get('load_path')
#         if load_path_G is not None:
#             self.load_network(self.net_G, load_path_G)
#             if self.verbose:
#                 self.logger.info('Load generator from: {}'.format(load_path_G))

#     def config_training(self):
#         # set criterion
#         self.set_criterion()

#         # set optimizer
#         self.optim_G = optim.Adam(
#             self.net_G.parameters(),
#             lr=self.opt['train']['generator']['lr'],
#             weight_decay=self.opt['train']['generator'].get('weight_decay', 0),
#             betas=(
#                 self.opt['train']['generator'].get('beta1', 0.9),
#                 self.opt['train']['generator'].get('beta2', 0.999)))

#         # set lr schedule
#         self.sched_G = define_lr_schedule(
#             self.opt['train']['generator'].get('lr_schedule'), self.optim_G)

#     def set_criterion(self):
#         # pixel criterion
#         self.pix_crit = define_criterion(
#             self.opt['train'].get('pixel_crit'))

#         # warping criterion
#         self.warp_crit = define_criterion(
#             self.opt['train'].get('warping_crit'))
        
#         # feature criterion
#         self.feat_crit = define_criterion(
#             self.opt['train'].get('feature_crit'))
#         if self.feat_crit is not None:  # load feature extractor
#             feature_layers = self.opt['train']['feature_crit'].get(
#                 'feature_layers', [8, 17, 26, 35])
#             self.net_F = VGGFeatureExtractor(feature_layers).to(self.device)

#         # flow & mask criterion
#         self.flow_crit = define_criterion(
#             self.opt['train'].get('flow_crit'))

#         # ping-pong criterion
#         self.pp_crit = define_criterion(
#             self.opt['train'].get('pingpong_crit'))

#     def extract_windows(self, data, window_size, stride):
#         """Extract sliding windows from a sequence
        
#         Args:
#             data: Input data with shape [B, T, C, H, W]
#             window_size: Size of each window
#             stride: Step size between windows
            
#         Returns:
#             List of windows, each with shape [B, window_size, C, H, W]
#         """
#         batch_size, seq_len, *_ = data.size()
#         windows = []
        
#         for i in range(0, seq_len - window_size + 1, stride):
#             window = data[:, i:i+window_size, ...]
#             windows.append(window)
            
#         return windows

#     def train(self, data):
#         """ Function of mini-batch training using sliding window approach

#             Parameters:
#                 :param data: a batch of training data (lr & gt) in shape ntchw
#         """

#         # ------------ prepare data ------------ #
#         lr_data, gt_data = data['lr'], data['gt']

#         # ------------ clear optim ------------ #
#         self.net_G.train()
#         self.optim_G.zero_grad()

#         # ------------ process with sliding window ------------ #
#         # Extract windows for both lr and gt data
#         lr_windows = self.extract_windows(lr_data, self.window_size, self.window_stride)
#         gt_windows = self.extract_windows(gt_data, self.window_size, self.window_stride)
        
#         # Process each window
#         loss_G = 0
#         self.log_dict = OrderedDict()
        
#         # Initialize cumulative losses
#         cum_loss_pix_G = 0
#         cum_loss_warp_G = 0
#         cum_loss_feat_G = 0
#         cum_loss_pp_G = 0
        
#         # Process each window independently
#         for lr_window, gt_window in zip(lr_windows, gt_windows):
#             # Forward pass for this window
#             net_G_output_dict = self.net_G.forward_window(lr_window)
#             hr_window = net_G_output_dict['hr_data']
            
#             # Pixel loss
#             pix_w = self.opt['train']['pixel_crit'].get('weight', 1)
#             loss_pix_G = pix_w * self.pix_crit(hr_window, gt_window)
#             loss_G += loss_pix_G
#             cum_loss_pix_G += loss_pix_G.item()
            
#             # Warping loss (if enabled)
#             if self.warp_crit is not None and 'lr_flow' in net_G_output_dict:
#                 # For sliding window, we compute warping between adjacent frames
#                 for i in range(1, self.window_size):
#                     lr_curr = net_G_output_dict.get('lr_frames', [])
#                     if len(lr_curr) > i:
#                         lr_prev = lr_curr[i-1]
#                         lr_curr_frame = lr_curr[i]
#                         lr_flow = net_G_output_dict.get('lr_flows', [])[i-1]
#                         lr_warp = net_utils.backward_warp(lr_prev, lr_flow)
                        
#                         warp_w = self.opt['train']['warping_crit'].get('weight', 1)
#                         loss_warp_G = warp_w * self.warp_crit(lr_warp, lr_curr_frame)
#                         loss_G += loss_warp_G
#                         cum_loss_warp_G += loss_warp_G.item()
            
#             # Feature loss
#             if self.feat_crit is not None:
#                 _, t, c, gt_h, gt_w = gt_window.size()
#                 hr_merge = hr_window.view(-1, c, gt_h, gt_w)
#                 gt_merge = gt_window.view(-1, c, gt_h, gt_w)
                
#                 hr_feat_lst = self.net_F(hr_merge)
#                 gt_feat_lst = self.net_F(gt_merge)
#                 loss_feat_G = 0
#                 for hr_feat, gt_feat in zip(hr_feat_lst, gt_feat_lst):
#                     loss_feat_G += self.feat_crit(hr_feat, gt_feat.detach())
                
#                 feat_w = self.opt['train']['feature_crit'].get('weight', 1)
#                 loss_feat_G = feat_w * loss_feat_G
#                 loss_G += loss_feat_G
#                 cum_loss_feat_G += loss_feat_G.item()
                
#             # Ping-pong loss
#             if self.pp_crit is not None:
#                 hr_data_fw = hr_window[:, :self.window_size - 1, ...]
#                 hr_data_bw = hr_window[:, 1:, ...].flip(1)
                
#                 pp_w = self.opt['train']['pingpong_crit'].get('weight', 1)
#                 loss_pp_G = pp_w * self.pp_crit(hr_data_fw, hr_data_bw)
#                 loss_G += loss_pp_G
#                 cum_loss_pp_G += loss_pp_G.item()
        
#         # Average the losses over all windows
#         num_windows = len(lr_windows)
#         if num_windows > 0:
#             self.log_dict['l_pix_G'] = cum_loss_pix_G / num_windows
#             if self.warp_crit is not None:
#                 self.log_dict['l_warp_G'] = cum_loss_warp_G / num_windows
#             if self.feat_crit is not None:
#                 self.log_dict['l_feat_G'] = cum_loss_feat_G / num_windows
#             if self.pp_crit is not None:
#                 self.log_dict['l_pp_G'] = cum_loss_pp_G / num_windows
        
#         # Optimize
#         loss_G.backward()
#         self.optim_G.step()

#     def infer(self, lr_data):
#         """ Function of inference using sliding window approach

#             Parameters:
#                 :param lr_data: a rgb video sequence with shape thwc
#                 :return: a rgb video sequence with type np.uint8 and shape thwc
#         """
#         print(lr_data.size())

#         # Eval mode
#         self.net_G.eval()

#         # canonicalize
#         lr_data = data_utils.canonicalize(lr_data)  # to torch.FloatTensor

#         # temporal padding to handle boundary conditions
#         lr_data, n_pad_front = self.pad_sequence(lr_data)
        
#         # Get dimensions
#         t, c, h, w = lr_data.size()
        
#         # Initialize output tensor
#         hr_seq = torch.zeros((t, c, h * self.scale_factor, w * self.scale_factor), 
#                            device=self.device)
        
#         # Process with overlapping windows
#         overlap_count = torch.zeros(t, device=self.device)
        
#         # Use half window overlap for better results
#         stride = max(1, self.window_size // 2)
        
#         # Process each window
#         for i in range(0, t - self.window_size + 1, stride):
#             # Extract window
#             window = lr_data[i:i+self.window_size].unsqueeze(0)  # Add batch dimension
            
#             # Process window
#             with torch.no_grad():
#                 window_output = self.net_G.forward_window(window)
#                 hr_window = window_output['hr_data'].squeeze(0)  # Remove batch dimension
            
#             # Add to output with overlap handling
#             for j in range(self.window_size):
#                 hr_seq[i+j] += hr_window[j]
#                 overlap_count[i+j] += 1
        
#         # Average overlapping regions
#         for i in range(t):
#             if overlap_count[i] > 0:
#                 hr_seq[i] /= overlap_count[i]
        
#         # Remove padding
#         hr_seq = hr_seq[n_pad_front:, ...]

#         # Train mode
#         self.net_G.train()

#         return hr_seq

#     def save(self, current_iter):
#         self.save_network(self.net_G, 'G', current_iter)


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
        if load_path_G is not None:
            self.load_network(self.net_G, load_path_G)
            if self.verbose:
                self.logger.info('Load generator from: {}'.format(load_path_G))

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
        
        # feature criterion
        self.feat_crit = define_criterion(
            self.opt['train'].get('feature_crit'))
        if self.feat_crit is not None:  # load feature extractor
            feature_layers = self.opt['train']['feature_crit'].get(
                'feature_layers', [8, 17, 26, 35])
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

        # warping loss
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

        # feature (feat) loss
        if self.feat_crit is not None:
            _, _, c, gt_h, gt_w = gt_data.size()
            hr_merge = hr_data.view(-1, c, gt_h, gt_w)
            gt_merge = gt_data.view(-1, c, gt_h, gt_w)

            hr_feat_lst = self.net_F(hr_merge)
            gt_feat_lst = self.net_F(gt_merge)
            loss_feat_G = 0
            for hr_feat, gt_feat in zip(hr_feat_lst, gt_feat_lst):
                loss_feat_G += self.feat_crit(hr_feat, gt_feat.detach())

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
        print(lr_data.size())

        # Eval mode
        self.net_G.eval()

        # canonicalize
        lr_data = data_utils.canonicalize(lr_data)  # to torch.FloatTensor
        # lr_data = lr_data.permute(0, 3, 1, 2)  # tchw

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
