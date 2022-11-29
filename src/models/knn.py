import torch
import torch.nn.functional as F
from tqdm import tqdm

import src.data.dataset as datasets
from src.engine import get_transform


# test using a knn monitor
def knn_test(net, memory_data_loader, test_data_loader, knn_k, knn_t):
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_num, feature_bank, feature_labels = 0.0, 0, [], []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            feature_labels.append(target.cuda(non_blocking=True))
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        # feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        feature_labels = torch.cat(feature_labels, dim=0).contiguous()
        # loop test data to predict the label by weighted knn search
        for data, target in tqdm(test_data_loader, desc='Evaluating on Test set'):
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()

    return total_top1 / total_num * 100

# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
# https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb#scrollTo=RI1Y8bSImD7N
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels

#----------------------------------------------------------------------------

class KNNEvaluator():
    def __init__(self,
        dataset_name,
        data_dir,
        input_scale,
        num_workers     = 4,
        knn_k           = 200,
        knn_t           = 0.1,
    ):
        # Hyperparameter configs
        self.knn_k  = knn_k
        self.knn_t  = knn_t

        val_transform = get_transform(input_scale, train=False)
        memory_dataset = datasets.__dict__[dataset_name](data_dir, 'train_knn', val_transform)
        self.memory_loader = torch.utils.data.DataLoader(
            memory_dataset, batch_size=256, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        val_dataset = datasets.__dict__[dataset_name](data_dir, 'val', val_transform)
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=256, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
    
    def evaluate(self, encoder):
        return knn_test(encoder, self.memory_loader, self.val_loader, self.knn_k, self.knn_t)

#----------------------------------------------------------------------------
