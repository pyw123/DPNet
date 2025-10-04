from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmpretrain.models.losses.prot_loss import diversity_loss
import torch

@HOOKS.register_module()
class PrototypeDiversityHook(Hook):
    def __init__(self, loss_weight=1.0):
        self.loss_weight = loss_weight

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        # 递归查找所有含有prothook的block
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module  # 兼容DDP
        found = False
        for name, module in model.named_modules():
            if hasattr(module, 'prothook') and 'refined_prototypes' in module.prothook:
                prototypes = module.prothook['refined_prototypes']  # (b, h, s, d)
                loss = diversity_loss(prototypes) * self.loss_weight
                found = True
                # 将损失加到runner的loss中
                if isinstance(outputs, dict) and 'loss' in outputs:
                    outputs['loss'] = outputs['loss'] + loss
                elif isinstance(outputs, dict):
                    outputs['loss'] = loss
                else:
                    runner.logger.warning('outputs 不是dict，无法加多样性损失')
        if not found:
            runner.logger.warning('未找到 refined_prototypes，未计算多样性损失')
