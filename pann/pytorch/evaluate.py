from sklearn import metrics

from pytorch_utils import forward   #把generator弄成一个字典
import numpy as np


def calculate_accuracy(y_true, y_score):                                #添加
     N = y_true.shape[0]
     accuracy = np.sum(np.argmax(y_true, axis=-1) == np.argmax(y_score, axis=-1)) / N
     return accuracy


class Evaluator(object):
    def __init__(self, model):
        """Evaluator.
            评估
        Args:
          model: object
        """
        self.model = model
        
    def evaluate(self, data_loader):
        """Forward evaluation data and calculate statistics.
            转发评估数据并计算统计数据。
        Args:
          data_loader: object

        Returns:
          statistics: dict, 
              {'average_precision': (classes_num,), 'auc': (classes_num,)}
        """

        # Forward
        output_dict = forward(
            model=self.model, 
            generator=data_loader, 
            return_target=True)

        clipwise_output = output_dict['clipwise_output']    # (audios_num, classes_num)
        target = output_dict['target']    # (audios_num, classes_num)

        average_precision = metrics.average_precision_score(
            target, clipwise_output, average=None)              #根据预测分数计算平均精度 (AP)

        auc = metrics.roc_auc_score(target, clipwise_output, average=None)      #根据预测分数计算ROC AUC下的面积

        cm = metrics.confusion_matrix(np.argmax(target, axis=-1), np.argmax(clipwise_output, axis=-1), labels=None)

        accuracy = calculate_accuracy(target, clipwise_output)

        statistics = {'average_precision': average_precision, 'auc': auc, 'accuracy': accuracy}

        return statistics

    #输出平均精度AP和AUC的面积