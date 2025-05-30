from utils.net_utils import get_upsampling_func


def define_generator(opt):
    net_G_opt = opt['model']['generator']

    if net_G_opt['name'].lower() == 'srnet':
        from .srnet import SRNet
        upsample_fn = get_upsampling_func(mode=net_G_opt['upsample_func'])
        net_G = SRNet(
            in_nc=net_G_opt['in_nc'],
            out_nc=net_G_opt['out_nc'],
            nf=net_G_opt['nf'],
            nb=net_G_opt['nb'],
            upsample_func=upsample_fn,
            scale=opt['scale'],
            transp_conv=net_G_opt['transp_conv'])
    elif net_G_opt['name'].lower() == 'frnet':  # frame-recurrent generator
        from .fr_net import FRNet
        net_G = FRNet(
            in_nc=net_G_opt['in_nc'],
            out_nc=net_G_opt['out_nc'],
            nf=net_G_opt['nf'],
            nb=net_G_opt['nb'],
            upsampling_fn=net_G_opt['upsample_func'],
            transp_conv=net_G_opt['transp_conv'],
            scale=opt['scale'],
            shallow_feat_res=net_G_opt.get('shallow_feat_res', False),
            with_tsa=net_G_opt.get('with_tsa', False))   
    elif net_G_opt['name'].lower() == 'dcnvsr':  # Deformable conv VSR generator
        from .dcn_net import DcnVSR
        net_G = DcnVSR(
            in_channels=net_G_opt['in_nc'],
            out_channels=net_G_opt['out_nc'],
            num_feat=net_G_opt['nf'],
            num_reconstruct_block=net_G_opt['nb'],
            num_deform_blocks=net_G_opt['num_deform_groups'],
            num_frames=net_G_opt['win_size'],
            upsampling_fn=net_G_opt['upsample_func'],
            shallow_feat_res=net_G_opt.get('shallow_feat_res', False),
            with_tsa=net_G_opt.get('with_tsa', False))       
    elif net_G_opt['name'].lower() == 'edvr':
        from .edvr_net import EDVRNet
        net_G = EDVRNet(
            in_channels=net_G_opt['in_nc'],
            out_channels=net_G_opt['out_nc'],
            num_feat=net_G_opt['nf'],
            num_frames=net_G_opt.get('num_frames', 5),
            deformable_groups=net_G_opt.get('def_groups', 8),
            num_extract_block=net_G_opt.get('num_extract', 5),
            num_reconstruct_block=net_G_opt['nb'],
            res_frame_idx=net_G_opt.get('res_frame_idx', None),
            with_tsa=net_G_opt['with_tsa'],
            upsample_func=net_G_opt['upsample_func'])
    else:
        raise ValueError('Unrecognized generator: {}'.format(
            net_G_opt['name']))

    return net_G


def define_discriminator(opt):
    net_D_opt = opt['model']['discriminator']

    if opt['dataset']['degradation']['type'] == 'BD':
        spatial_size = opt['dataset']['train']['gt_crop_size']
    else:  # BI
        spatial_size = opt['dataset']['train']['gt_crop_size']

    if net_D_opt['name'].lower() == 'stnet':  # spatio-temporal discriminator
        from .discriminator_nets import SpatioTemporalDiscriminator
        net_D = SpatioTemporalDiscriminator(
            in_nc=net_D_opt['in_nc'],
            spatial_size=spatial_size,
            tempo_range=net_D_opt['tempo_range'],
            spectral_norm=net_D_opt.get('spectral_norm', False))

    elif net_D_opt['name'].lower() == 'snet':  # spatial discriminator
        from .discriminator_nets import SpatialDiscriminator
        net_D = SpatialDiscriminator(
            in_nc=net_D_opt['in_nc'],
            spatial_size=spatial_size,
            use_cond=net_D_opt['use_cond'])

    else:
        raise ValueError('Unrecognized discriminator: {}'.format(
            net_D_opt['name']))

    return net_D
