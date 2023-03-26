"""Classification Performance"""
import torch


class PerfClassif():
    """Classification Performance"""
    @classmethod
    def gen_fun_acc(cls, dim_target):
        """
        :param dim_target: class/domain label embeding dimension
        """
        def fun_acc(list_vec_preds, list_vec_labels):
            """
            :param list_vec_preds: list of batches
            """
            assert len(list_vec_preds) > 0
            correct_count = 0
            obs_count = 0
            for pred, label in zip(list_vec_preds, list_vec_labels):
                correct_count += torch.sum(torch.sum(pred == label, dim=1) == dim_target)
                obs_count += pred.shape[0]  # batch size
            if isinstance(correct_count, int):
                acc = (correct_count) / obs_count
            else:
                acc = (correct_count.float()) / obs_count
            # AttributeError: 'int' object has no attribute 'float'
            # reason: batchsize is too big
            return acc
        return fun_acc

    @classmethod
    def cal_acc(cls, model, loader_te, device):
        """
        :param model:
        :param loader_te:
        :param device: for final test, GPU can be used
        """
        model.eval()
        model_local = model.to(device)
        fun_acc = cls.gen_fun_acc(model_local.dim_y)
        list_vec_preds, list_vec_labels = cls.get_list_pred_target(
            model_local, loader_te, device)
        accuracy_y = fun_acc(list_vec_preds, list_vec_labels)
        acc_y = accuracy_y.cpu().numpy().item()
        return acc_y

    @classmethod
    def get_list_pred_target(cls, model_local, loader_te, device):
        """
        isolate function to check if prediction persist each time loader is went through
        """
        list_vec_preds, list_vec_labels = [], []
        with torch.no_grad():
            for _, (x_s, y_s, *_) in enumerate(loader_te):
                x_s, y_s = x_s.to(device), y_s.to(device)
                pred, *_ = model_local.infer_y_vpicn(x_s)
                list_vec_preds.append(pred)
                list_vec_labels.append(y_s)
        return list_vec_preds, list_vec_labels
