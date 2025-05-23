import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_gradient_penalty(D, real_samples, fake_samples, lambda_gp=10, device='cuda'):
        """
        Computes the gradient penalty for WGAN-GP.

        Args:
            D (nn.Module): The discriminator (or critic).
            real_samples (Tensor): Real images [B, C, H, W].
            fake_samples (Tensor): Fake images [B, C, H, W].
            device (str): Device to perform computation on.
            lambda_gp (float): Gradient penalty coefficient.

        Returns:
            Tensor: Scalar gradient penalty loss.
        """
        batch_size = real_samples.size(0)

        # Interpolate between real and fake
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

        # Forward pass
        d_interpolates = D(interpolates)

        # If D returns [B, 1], flatten to [B]
        if d_interpolates.ndim > 1:
            d_interpolates = d_interpolates.view(-1)

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates, device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)  # L2 norm

        gp = lambda_gp * ((gradient_norm - 1) ** 2).mean()
        return gp


class VanillaGANLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(VanillaGANLoss, self).__init__()
        self.crit = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, input, status):
        """
            :param status: boolean, True/False
        """
        target = torch.empty_like(input).fill_(int(status))
        loss = self.crit(input, target)
        return loss


class WGANLoss(nn.Module):
    def __init__(self, lambda_gp=10.0):
        super(WGANLoss, self).__init__()
        self.lambda_gp = lambda_gp

    def forward(self, input, status):
        """
        :param input: Discriminator output
        :param status: 1 if real, 0 if fake
        :return: Wasserstein loss
        """
        # Real -> hope for high scores → maximize D(x_real) → -D(x_real)
        # Fake -> hope for low scores → minimize D(x_fake) → D(x_fake)
        return -input.mean() if status == 1 else input.mean()

class CharbonnierLoss(nn.Module):
    """ Charbonnier Loss (robust L1)
    """

    def __init__(self, eps=1e-6, reduction='sum'):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps)

        if self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'mean':
            loss = torch.mean(loss)
        else:
            raise NotImplementedError
        return loss


class CosineSimilarityLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(CosineSimilarityLoss, self).__init__()
        self.eps = eps

    def forward(self, input, target):
        diff = F.cosine_similarity(input, target, dim=1, eps=self.eps)
        loss = 1.0 - diff.mean()

        return loss
