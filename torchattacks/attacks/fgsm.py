import torch
import torch.nn as nn

from ..attack import Attack
from yolo_adv.utils import YOLOv8DetectionLoss

class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=8/255)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, yolo=False, eps=8 / 255):
        super().__init__("FGSM", yolo, model)
        self.eps = eps
        self.supported_mode = ["default", "targeted"]
        if self.yolo:
            self.loss_obj = YOLOv8DetectionLoss(model, max_steps=1)

    def forward(self, images, labels, bboxes=None):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        if not self.yolo:
            loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        if not self.yolo:
            outputs = self.get_logits(images)

        # Calculate loss
        if self.targeted:
            if self.yolo:
                cost = -self.loss_obj.compute_loss(images, target_labels, bboxes, 0, requires_grad=True)
            else:
                cost = -loss(outputs, target_labels)
        else:
            if self.yolo:
                cost = self.loss_obj.compute_loss(images, labels, bboxes, 0, requires_grad=True)
            else:      
                cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(
            cost, images, retain_graph=False, create_graph=False
        )[0]

        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        if self.yolo:
            return self.loss_obj.losses, adv_images
        else:
            return adv_images
