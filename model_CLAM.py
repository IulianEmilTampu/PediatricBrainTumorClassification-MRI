import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
import numpy as np


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


class Attn_Net(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [nn.Linear(L, D), nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""


class CLAM_SB(nn.Module):
    def __init__(
        self,
        instance_feature_size:int=1024, # IET Added to account for the instance features of different sizes
        gate=True,
        size_arg="small",
        dropout=False,
        k_sample=8,
        n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(),
        subtyping=False,
    ):
        super(CLAM_SB, self).__init__()
        self.size_dict = {"small": [instance_feature_size, 512, 256], "big": [instance_feature_size, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(
                L=size[1], D=size[2], dropout=dropout, n_classes=1
            )
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        initialize_weights(self)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    # instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(
        self,
        h,
        label=None,
        instance_eval=False,
        return_features=False,
        attention_only=False,
    ):
        device = h.device
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(
                label, num_classes=self.n_classes
            ).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(
                            A, h, classifier
                        )
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h)
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        if instance_eval:
            results_dict = {
                "instance_loss": total_inst_loss,
                "inst_labels": np.array(all_targets),
                "inst_preds": np.array(all_preds),
            }
        else:
            results_dict = {}
        if return_features:
            results_dict.update({"features": M})
        return logits, Y_prob, Y_hat, A_raw, results_dict


class CLAM_MB(CLAM_SB):
    def __init__(
        self,
        instance_feature_size: int = 1024,  # IET Added to account for the instance features of different sizes
        gate=True,
        size_arg="small",
        dropout=False,
        k_sample=8,
        n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(),
        subtyping=False,
    ):
        nn.Module.__init__(self)
        self.size_dict = {
            "small": [instance_feature_size, 512, 256],
            "big": [instance_feature_size, 512, 384],
        }
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(
                L=size[1], D=size[2], dropout=dropout, n_classes=n_classes
            )
        else:
            attention_net = Attn_Net(
                L=size[1], D=size[2], dropout=dropout, n_classes=n_classes
            )
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [
            nn.Linear(size[1], 1) for i in range(n_classes)
        ]  # use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        initialize_weights(self)

    def forward(
        self,
        h,
        label=None,
        instance_eval=False,
        return_features=False,
        attention_only=False,
    ):
        device = h.device
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(
                label, num_classes=self.n_classes
            ).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                # print(f"CLAM_MB forward: label shape {label.shape}")
                # print(f"CLAM_MB forward: inst_labels shape {inst_labels.shape}")
                # print(f"CLAM_MB forward: inst_labels at {i} {inst_labels[i]}")
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(
                            A[i], h, classifier
                        )
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h)
        logits = torch.empty(1, self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        if instance_eval:
            results_dict = {
                "instance_loss": total_inst_loss,
                "inst_labels": np.array(all_targets),
                "inst_preds": np.array(all_preds),
            }
        else:
            results_dict = {}
        if return_features:
            results_dict.update({"features": M})
        return logits, Y_prob, Y_hat, A_raw, results_dict


def train_loop_clam(
    epoch, model, loader, optimizer, n_classes, bag_weight, writer=None, loss_fn=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)

    train_loss = 0.0
    train_error = 0.0
    train_inst_loss = 0.0
    inst_count = 0

    print("\n")
    for batch_idx, (data, label) in enumerate(loader):
        # print(f"training loop: data shape {data.shape}")
        # print(f"training loop: label shape {data.shape}")
        data, label = data.to(device), label.to(device)
        logits, Y_prob, Y_hat, _, instance_dict = model(
            data, label=label, instance_eval=True
        )

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict["instance_loss"]
        inst_count += 1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value

        total_loss = bag_weight * loss + (1 - bag_weight) * instance_loss

        inst_preds = instance_dict["inst_preds"]
        inst_labels = instance_dict["inst_labels"]
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print(
                "batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, ".format(
                    batch_idx, loss_value, instance_loss_value, total_loss.item()
                )
                + "label: {}, bag_size: {}".format(label.item(), data.size(0))
            )

        error = calculate_error(Y_hat, label)
        train_error += error

        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    if inst_count > 0:
        train_inst_loss /= inst_count
        print("\n")
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print(
                "class {} clustering acc {}: correct {}/{}".format(
                    i, acc, correct, count
                )
            )

    print(
        "Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}".format(
            epoch, train_loss, train_inst_loss, train_error
        )
    )
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print("class {}: acc {}, correct {}/{}".format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar("train/class_{}_acc".format(i), acc, epoch)

    if writer:
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/error", train_error, epoch)
        writer.add_scalar("train/clustering_loss", train_inst_loss, epoch)


def validate_clam(
    cur,
    epoch,
    model,
    loader,
    n_classes,
    early_stopping=None,
    writer=None,
    loss_fn=None,
    results_dir=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.0
    val_error = 0.0

    val_inst_loss = 0.0
    val_inst_acc = 0.0
    inst_count = 0

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            # print(f"evaluate clam: data shape {data.shape}")
            # print(f"evaluate clam: data shape {label.shape}")
            logits, Y_prob, Y_hat, _, instance_dict = model(
                data, label=label, instance_eval=True
            )
            # print(f"evaluate clam: Y_hat shape {Y_hat.shape}")
            acc_logger.log(Y_hat, label)

            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict["instance_loss"]

            inst_count += 1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict["inst_preds"]
            inst_labels = instance_dict["inst_labels"]
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()

            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float("nan"))

        auc = np.nanmean(np.array(aucs))

    print(
        "\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}".format(
            val_loss, val_error, auc
        )
    )
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(n_classes):
            acc, correct, count = inst_logger.get_summary(i)
            print(
                "class {} clustering acc {}: correct {}/{}".format(
                    i, acc, correct, count
                )
            )

    if writer:
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/auc", auc, epoch)
        writer.add_scalar("val/error", val_error, epoch)
        writer.add_scalar("val/inst_loss", val_inst_loss, epoch)

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print("class {}: acc {}, correct {}/{}".format(i, acc, correct, count))

        if writer and acc is not None:
            writer.add_scalar("val/class_{}_acc".format(i), acc, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(
            epoch,
            val_loss,
            model,
            ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)),
        )

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


class Accuracy_Logger(object):
    """Accuracy logger"""

    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += Y_hat == Y

    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]

        if count == 0:
            acc = None
        else:
            acc = float(correct) / count

        return acc, correct, count


def calculate_error(Y_hat, Y):
    error = 1.0 - Y_hat.float().eq(Y.float()).float().mean().item()

    return error
