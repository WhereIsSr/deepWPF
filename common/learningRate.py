import logging


def adjust_learning_rate(optimizer, epoch, adjust_type, lr_begin):
    """
    Desc:
        Adjust learning rate
    """
    lr_adjust = {}
    if adjust_type == 'type1':
        # learning_rate = 0.5^{epoch-1}
        lr_adjust = {epoch: lr_begin * (0.50 ** (epoch - 1))}
    elif adjust_type == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        logging.info(f"adjust learning rate to {lr}")
