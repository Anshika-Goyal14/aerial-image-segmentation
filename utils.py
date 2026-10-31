import torch 
import torch.nn as nn
import torch.nn.functional as F 

def compute_iou(pred, target,num_classes =5): #intersection over union to get the correct pixels 
    ious= []
    for c in range(num_classes):
        pred_c = (pred==c)
        target_c = (target==c)
        intersection = (pred_c & target_c).sum()
        union  = (pred_c | target_c).sum()

        if union == 0 :
            ious.append(float('nan'))

        else : 
            ious.append(intersection.item()/union.item())
    return ious


class DiceLoss(nn.Module):
    def __init__(self, num_classes = 5, smooth = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth 

    def forward(self, outputs, targets):
        probs = F.softmax(outputs, dim=1) #scores into probabilities

        #one hot is used so that they match the shape of probs 
        targets_onehot = F.one_hot(targets, self.num_classes)# F.one_hot gives B,H,W,5 
        targets_onehot = targets_onehot.permute(0,3,1,2).float()#B,5,H,W

        #sum over batch and spatial dims 0,2,3
        intersection = (probs* targets_onehot).sum(dim=(0,2,3))
        denominator = probs.sum(dim=(0,2,3))+targets_onehot.sum(dim=(0,2,3))

        dice = (2*intersection + self.smooth)/(denominator + self.smooth)

        return 1- dice.mean() #average over classes =, then change high overlap to low loss



if __name__ == "__main__":
    pred = torch.tensor([1,1,0,0,1,0]) #testing iou with 6 pixels
    target = torch.tensor([1,0,0,1,1,0])
    print(compute_iou(pred,target,num_classes = 3))

    outputs = torch.randn(2,5,64,64) #testing dice loss
    targets = torch.randint(0,5,(2,64,64))
    print(DiceLoss()(outputs, targets)) #should sit between 0 and 1 

