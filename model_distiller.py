import torch
import torch.nn as nn

class ICKDLoss(nn.Module):
    """Inter-Channel Correlation"""

    def __init__(self):
        super(ICKDLoss, self).__init__()

    def forward(self, g_s, g_t):
        loss = [self.batch_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]
        return loss


    def batch_loss(self, f_s, f_t):
        bsz, ch = f_s.shape[0], f_s.shape[1]

        f_s = f_s.view(bsz, ch, -1)
        f_t = f_t.view(bsz, ch, -1)

        emd_s = torch.bmm(f_s, f_s.permute(0, 2, 1))
        emd_s = torch.nn.functional.normalize(emd_s, dim=2)

        emd_t = torch.bmm(f_t, f_t.permute(0, 2, 1))
        emd_t = torch.nn.functional.normalize(emd_t, dim=2)

        G_diff = emd_s - emd_t
        loss = (G_diff * G_diff).view(bsz, -1).sum() / (ch * bsz)
        return loss


class DistillerNet(nn.Module):
    def __init__(self): #student_feature  teacher_feature
        super(DistillerNet, self).__init__()
        self.distiller_loss = ICKDLoss().cuda()

    def forward(self, sf:list, tf:list):
        los_list = []

        sf[0] = self.channel_relation(sf[0], tf[0])
        sf[1] = self.channel_relation(sf[1], tf[1])
        loss1 = self.distiller_loss([sf[0]], [tf[0]])[0]
        los_list.append(loss1)
        loss2 = self.distiller_loss([sf[1]], [tf[1]])[0]
        los_list.append(loss2)

        return los_list
    
    def channel_relation(self,sf,tf):
        bsz, ch = sf.shape[0], sf.shape[1]
        sf = sf.view(bsz, ch, -1)
        tf = tf.view(bsz, ch, -1)

        emd_s = torch.bmm(sf, sf.permute(0, 2, 1))
        emd_s = torch.nn.functional.normalize(emd_s, dim=2)

        emd_t = torch.bmm(tf, tf.permute(0, 2, 1))
        emd_t = torch.nn.functional.normalize(emd_t, dim=2)

        emd_st = torch.bmm(emd_s, emd_t)
        emd_st = nn.Softmax(dim=-1)(emd_st)

        emd_s = torch.bmm(emd_st, sf)

        emd_s = (sf + emd_s).view(bsz,ch,-1)
        return emd_s






