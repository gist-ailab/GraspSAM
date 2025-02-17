class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches-1) + ']'
    
class TrainProgress():
    
    def __init__(self, dataloader, epoch):
        self.losses = AverageMeter('Loss', ':.4f')
        self.p_loss = AverageMeter('pos_loss', ':.4f')
        self.c_loss = AverageMeter('cos_loss', ':.4f')
        self.s_loss = AverageMeter('sin_loss', ':.4f')
        self.w_loss = AverageMeter('width_loss', ':.4f')
        
        self.m_loss = AverageMeter('mask_loss', ':.4f')

        self.progress = ProgressMeter(
            len(dataloader),
            [self.losses, self.p_loss, self.c_loss, self.s_loss, self.w_loss, self.m_loss],
            prefix="Epoch: [{}]".format(epoch))
        

    def progress_update(self, lossd, batch_size):
        
        losses = lossd["loss"].item()
        
        p_loss = lossd["g_losses"]["p_loss"].item()   
        cos_loss = lossd["g_losses"]["cos_loss"].item()
        sin_loss = lossd["g_losses"]["sin_loss"].item()
        width_loss = lossd["g_losses"]["width_loss"].item()
        
        mask_loss = lossd["mask_loss"].item()

        self.losses.update(losses, batch_size)
        self.p_loss.update(p_loss, batch_size)
        self.c_loss.update(cos_loss, batch_size)
        self.s_loss.update(sin_loss, batch_size)
        self.w_loss.update(width_loss, batch_size)
        self.m_loss.update(mask_loss, batch_size)


        