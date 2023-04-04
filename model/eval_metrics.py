from asyncio.log import logger
import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
import logging

logger = logging.getLogger(__name__)

def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " %(key, src_dict[key]) 
    return dst_str

def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = (test_truth_emo > 0)
    predicted_label = (test_preds_emo > 0)
    tp = float(np.sum((true_label==1) & (predicted_label==1)))
    tn = float(np.sum((true_label==0) & (predicted_label==0)))
    p = float(np.sum(true_label==1))
    n = float(np.sum(true_label==0))

    return (tp * (n/p) +tn) / (2*n)

def eval_mosei_senti(results, truths, exclude_zero=False, bTest=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)

    # f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    binary_truth_non0 = test_truth[non_zeros] > 0
    binary_preds_non0 = test_preds[non_zeros] > 0
    f_score_non0 = f1_score(binary_truth_non0, binary_preds_non0, average='weighted')
    acc_2_non0 = accuracy_score(binary_truth_non0, binary_preds_non0)

    binary_truth_has0 = test_truth >= 0
    binary_preds_has0 = test_preds >= 0
    acc_2 = accuracy_score(binary_truth_has0, binary_preds_has0)
    f_score = f1_score(binary_truth_has0, binary_preds_has0, average='weighted')

    logger.info("-" * 50)
    logger.info("MAE: {}".format(mae))
    logger.info("Correlation Coefficient: {}".format(corr))
    logger.info("mult_acc_7: {}".format(mult_a7))
    logger.info("mult_acc_5: {}".format(mult_a5))
    logger.info("F1 score all/non0: {}/{} over {}/{}".format(np.round(f_score, 4), np.round(f_score_non0, 4),
                                                    binary_truth_has0.shape[0], binary_truth_non0.shape[0]))
    logger.info("Accuracy_2 all/non0: {}/{}".format(np.round(acc_2, 4), np.round(acc_2_non0, 4)))

    logger.info("-" * 50)
    
    return acc_2_non0


def eval_mosi(results, truths, exclude_zero=False, bTest=False):
    return eval_mosei_senti(results, truths, exclude_zero, bTest)


def eval_ur_funny(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    binary_truth = (test_truth > 0.5)
    binary_preds = (test_preds > 0)
    accu=accuracy_score(binary_truth, binary_preds)

    loginfo=''
    loginfo+="Accuracy: "+str(accu)+'\n'
    loginfo+="-" * 50+'\n'
    return accu,loginfo


def eval_sims_regression(y_pred, y_true, bTest=False):
    test_preds = y_pred.view(-1).cpu().detach().numpy()
    test_truth = y_true.view(-1).cpu().detach().numpy()
    test_preds = np.clip(test_preds, a_min=-1., a_max=1.)
    test_truth = np.clip(test_truth, a_min=-1., a_max=1.)

    # two classes{[-1.0, 0.0], (0.0, 1.0]}
    ms_2 = [-1.01, 0.0, 1.01]
    test_preds_a2 = test_preds.copy()
    test_truth_a2 = test_truth.copy()
    for i in range(2):
        test_preds_a2[np.logical_and(test_preds > ms_2[i], test_preds <= ms_2[i+1])] = i
    for i in range(2):
        test_truth_a2[np.logical_and(test_truth > ms_2[i], test_truth <= ms_2[i+1])] = i

    # three classes{[-1.0, -0.1], (-0.1, 0.1], (0.1, 1.0]}
    ms_3 = [-1.01, -0.1, 0.1, 1.01]
    test_preds_a3 = test_preds.copy()
    test_truth_a3 = test_truth.copy()
    for i in range(3):
        test_preds_a3[np.logical_and(test_preds > ms_3[i], test_preds <= ms_3[i+1])] = i
    for i in range(3):
        test_truth_a3[np.logical_and(test_truth > ms_3[i], test_truth <= ms_3[i+1])] = i
    
    # five classes{[-1.0, -0.7], (-0.7, -0.1], (-0.1, 0.1], (0.1, 0.7], (0.7, 1.0]}
    ms_5 = [-1.01, -0.7, -0.1, 0.1, 0.7, 1.01]
    test_preds_a5 = test_preds.copy()
    test_truth_a5 = test_truth.copy()
    for i in range(5):
        test_preds_a5[np.logical_and(test_preds > ms_5[i], test_preds <= ms_5[i+1])] = i
    for i in range(5):
        test_truth_a5[np.logical_and(test_truth > ms_5[i], test_truth <= ms_5[i+1])] = i

    mae = np.mean(np.absolute(test_preds - test_truth)).astype(np.float64)   # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a2 = multiclass_acc(test_preds_a2, test_truth_a2)
    mult_a3 = multiclass_acc(test_preds_a3, test_truth_a3)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    f_score = f1_score(test_truth_a2, test_preds_a2, average='weighted')

    eval_results = {
        "Mult_acc_2": round(mult_a2, 4),
        "Mult_acc_3": round(mult_a3, 4),
        "Mult_acc_5": round(mult_a5, 4),
        "F1_score": round(f_score, 4),
        "MAE": round(mae, 4),
        "Corr": round(corr, 4), # Correlation Coefficient
    }

    logger.info("-" * 50)
    logger.info("MAE: {}".format(round(mae, 4)))
    logger.info("Correlation Coefficient: {}".format(round(corr, 4)))
    logger.info("mult_acc_5: {}".format(round(mult_a5, 4)))
    logger.info("mult_acc_3: {}".format(round(mult_a3, 4)))
    logger.info("mult_acc_2: {}".format(round(mult_a2, 4)))
    logger.info("F1_score: {}".format(round(f_score, 4)))
    logger.info("-" * 50)

    return mult_a2