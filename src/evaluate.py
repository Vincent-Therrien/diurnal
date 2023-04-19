"""
    RNA secondary prediction evaluation module.

    This module contains functions to evaluate RNA secondary predictions
    by comparing a predicted structure to a reference structure.

    Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    Affiliation: Département d'informatique, UQÀM
    File creation date: April 2023
    License: MIT
"""

# Evaluation based on paired / unpaired sensitivity.
def get_true_positive(prediction, reference, unpaired_symbol="."):
    """
    Compute the true positive value obtained by comparing two secondary
    structures.
    
    The true positive (TP) value is defined as the number of bases that
    are correclty predicted to be paired with another base. For example,
    in the following *demonstrative* secondary structures:
    - predicton: (..(((....
    - reference: (.....))))
    one paired base is correctly predicted, so the function returns 1.
    """
    tp = 0
    for i in range(min(len(prediction), len(reference))):
        if prediction[i] != unpaired_symbol and prediction[i] == reference[i]:
            tp += 1
    return tp

def get_true_negative(prediction, reference, unpaired_symbol="."):
    """
    Compute the true negative value obtained by comparing two secondary
    structures.
    
    The true negative (TN) value is defined as the number of bases that
    are correclty predicted to be unpaired.For example, in the following
    *demonstrative* secondary structures:
    - predicton: (..(((....
    - reference: (.....))))
    Two unpaired base are correctly predicted, so the function returns
    2.
    """
    tn = 0
    for i in range(min(len(prediction), len(reference))):
        if prediction[i] == unpaired_symbol and prediction[i] == reference[i]:
            tn += 1
    return tn

def get_false_positive(prediction, reference, unpaired_symbol="."):
    """
    Compute the false positive value obtained by comparing two secondary
    structures.
    
    The false positive (FP) value is defined as the number of bases that
    are predicted to be paired but that are actually unpaired. For
    example, in the following *demonstrative* secondary structures:
    - predicton: (..(((....
    - reference: (.....))))
    three bases are incorrectly predicted as paired, so the function
    returns 3.
    """
    fp = 0
    for i in range(min(len(prediction), len(reference))):
        if prediction[i] != unpaired_symbol and prediction[i] != reference[i]:
            fp += 1
    return fp

def get_false_negative(prediction, reference, unpaired_symbol="."):
    """
    Compute the false negative value obtained by comparing two secondary
    structures.
    
    The false negative (FN) value is defined as the number of bases that
    are predicted to be unpaired but that are actually paired. For
    example, in the following *demonstrative* secondary structures:
    - predicton: (..(((....
    - reference: (.....))))
    four bases are incorrectly predicted as unpaired, so the function
    returns 4.
    """
    fn = 0
    for i in range(min(len(prediction), len(reference))):
        if prediction[i] == unpaired_symbol and prediction[i] != reference[i]:
            fn += 1
    return fn

def get_sensitivity(prediction, reference, unpaired_symbol="."):
    """
    Compute the sensitivity value (SEN) obtained by comparing two
    secondary structures. The sensitivity is defined as: TP / (TP + FN).
    """
    tp = get_true_positive(prediction, reference, unpaired_symbol)
    fn = get_false_negative(prediction, reference, unpaired_symbol)
    if tp + fn:
        return tp / (tp + fn)
    else:
        return 0.0

def get_PPV(prediction, reference, unpaired_symbol="."):
    """
    Compute the positive predictive value (PPV) obtained by comparing
    two secondary structures. The PPV is defined as: TP / (TP + FP).
    """
    tp = get_true_positive(prediction, reference, unpaired_symbol)
    fp = get_false_positive(prediction, reference, unpaired_symbol)
    if tp + fp:
        return tp / (tp + fp)
    else:
        return 0.0

def get_sen_PPV_f1(prediction, reference, unpaired_symbol="."):
    """
    Compute the F1-score obtained by comparing two secondary structures.
    The f1-score is defined as: 2 * ((SEN*PPV) / (SEN+PPV)).
    """
    sen = get_sensitivity(prediction, reference, unpaired_symbol)
    ppv = get_PPV(prediction, reference, unpaired_symbol)
    if sen + ppv:
        f1 = 2 * ((sen*ppv) / (sen+ppv))
        return sen, ppv, f1
    else:
        return sen, ppv, 0.0
