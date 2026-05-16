class ProgLoss:
    """
    YOLO26-style progressive loss controller.

    For the public YOLO26n recipe direction:
        one2one is the main end-to-end branch.
        one2many gives dense auxiliary supervision.

    This class returns branch/loss weights used by YOLO26DetectionLoss.
    """

    def __init__(
        self,
        one2one_weight=1.0,
        one2many_weight=1.0,
        box_weight=7.5,
        class_weight=2.74,
        warmup_epochs=5,
    ):
        self.one2one_weight = one2one_weight
        self.one2many_weight = one2many_weight
        self.box_weight = box_weight
        self.class_weight = class_weight
        self.warmup_epochs = warmup_epochs

    def weights(self, epoch=None, total_epochs=None):
        progress = self._progress(epoch, total_epochs)

        # Early epochs: keep dense one2many branch strong.
        # Later epochs: keep one2one as the main branch.
        if epoch is None:
            one2many_scale = self.one2many_weight
        else:
            if epoch <= self.warmup_epochs:
                one2many_scale = self.one2many_weight
            else:
                one2many_scale = self.one2many_weight * (1.0 - 0.15 * progress)

        return {
            "one2one": self.one2one_weight,
            "one2many": max(one2many_scale, 0.25),
            "box": self.box_weight,
            "class": self.class_weight,
        }

    def _progress(self, epoch, total_epochs):
        if epoch is None or total_epochs is None or total_epochs <= 1:
            return 1.0

        progress = float(epoch - 1) / float(total_epochs - 1)
        return max(0.0, min(1.0, progress))