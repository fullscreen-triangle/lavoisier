import getopt
import numpy as np
import os
import sys

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.AllChem import GetHashedAtomPairFingerprintAsBitVect
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.metrics import *
from xgboost import XGBClassifier


def parse_input(filename):
    """
    parses tsv input file and generates RDKit molecules

    :param filename:
    :return: list of generated IDs, list of SMILES strings, list of RDKit molecules, activity (0 or 1)
    """
    if not os.path.isfile(filename):
        print(f"Path to input file does not point to a file:  {filename}", sys.stderr)
        exit(-1)
    orig_smiles = list()
    molecules = list()
    activity = list()
    ids = list()

    with open(filename, 'r') as f:
        for line in f:
            split = line.split('\t')
            if len(split) == 1:  # Catches the case of empty lines
                continue
            smile = split[0].replace('"', '')
            mol = Chem.MolFromSmiles(smile)
            class_label = int(split[1].split('\n')[0].rstrip())
            if mol:
                orig_smiles.append(smile.rstrip())
                molecules.append(Chem.MolFromSmiles(smile))
                if class_label == 1:
                    activity.append(class_label)
                else:
                    activity.append(0)
            else:
                print(f"Skipping molecule for SMILES {smile}, as RDKit could not parse it.")
        for i in range(1, len(molecules) + 1):  # generate IDs
            ids.append(i)
    # to np array
    activity = np.asarray(activity)

    return ids, orig_smiles, molecules, activity


def final_predictor(molecules_train, activity_train, molecules_test):
    """
    Final predictor of our project.
    Generates predictor and feature vectors
    Trains predictor and predicts test data

    :param molecules_train: list of RDKit molecules
    :param activity_train: list of class labels
    :param molecules_test: rdkit molecules for the prediction
    :return: vector with class labels (either 0 or 1)
    """

    # Default Classifier
    clf = XGBClassifier(random_state=1, n_jobs=-1)

    # Generate feature vector for the training data
    feature_vector_for_train = generate_initial_feature_vector(molecules_train, activity_train)
    feature_vector_train = select_most_important_features(clf, feature_vector_for_train, 103, activity_train,
                                                          feature_vector_for_train)

    # Generate feature vector for the data to be predicted
    feature_vector_test = generate_initial_feature_vector(molecules_train, activity_train, molecules_test)
    feature_vector_test = select_most_important_features(clf, feature_vector_for_train.copy(), 103, activity_train,
                                                         feature_vector_test)

    final_clf = XGBClassifier(random_state=1, n_jobs=-1, subsample=0.75, reg_lambda=2.0, reg_alpha=0.0,
                              n_estimators=150, min_child_weight=0.5, max_depth=8, learning_rate=0.2,
                              gamma=0.5, colsample_bytree=1.0)

    # train the classifier
    final_clf.fit(feature_vector_train, activity_train)

    # make prediction
    predictions = final_clf.predict(feature_vector_test)

    return predictions


def feature_fingerprint_morgan(molecules, radius, bits):
    """
    Generates circular morgan fingerprints for each molecule

    :param molecules: list of RDKit molecules
    :param radius: radius for morgan fingerprint generation
    :param bits: length of bit vector
    :return: list of fingerprints
    """
    feature_vector = []

    for mol in molecules:
        # radius 3 equals ecpf6 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4510302/
        string_bits = GetMorganFingerprintAsBitVect(mol, radius, nBits=bits).ToBitString()
        feature_vector.append(list(map(int, string_bits)))
    return feature_vector


def feature_fingerprint_atom_pair(molecules, bits):
    """
    Generates atom pair fingerprints for each molecules

    :param molecules: list of RDKit molecules
    :param bits: length of bit vector
    :return: list of fingerprints
    """
    feature_vector = []

    for mol in molecules:
        string_bits = GetHashedAtomPairFingerprintAsBitVect(mol, nBits=bits).ToBitString()
        feature_vector.append(list(map(int, string_bits)))
    return feature_vector


def generate_molecular_descriptors(molecules):
    """
    Generates molecular descriptors (RDKit) for each molecule

    :param molecules: list of RDKit molecules
    :return: list of list of descriptors for each molecule
    """
    names_descriptors = [x[0] for x in Descriptors._descList]
    my_desc_obj = MoleculeDescriptors.MolecularDescriptorCalculator(names_descriptors)
    feature_vector = [my_desc_obj.CalcDescriptors(x) for x in molecules]
    feature_vector = np.asarray(feature_vector)
    feature_vector[np.isnan(feature_vector)] = 0  # Replace NaN with 0
    return feature_vector


def generate_initial_feature_vector(molecules_1, activity, molecules_2=None):
    """
    Creates a feature vector based on RDKit molecular descriptors, Atom pair and Morgan Fingerprints
    Removes unimportant features for XGBClassifier. The importance of the feature is based on the molecules
    handed as molecules_1.
    If molecules_2 is not specified, the returned feature vector is generated for molecules_1.
    If molecules_2 is specified, the returned feature vector is generated for molecules_2

    :param molecules_1: list of RDKit molecules
    :param activity: list of class labels
    :param molecules_2: list of RDKit molecules to create feature vector from
    :return: feature vector
    """
    if molecules_2 is None:
        datasets = [molecules_1]
    else:
        datasets = [molecules_1, molecules_2]

    all_feature_vectors = []

    for d in datasets:
        # Generate feature vector
        feature_vector = generate_molecular_descriptors(d)

        # ECFP8 feature vector
        ecfp8 = np.asarray(feature_fingerprint_morgan(d, 4, 4096))

        # linear fingerprints feature vector (Atom pair=
        lin_fingerprint = np.asarray(feature_fingerprint_atom_pair(d, 4096))

        # combination
        combined_feature_vector = []
        for x, y, z in zip(ecfp8, feature_vector, lin_fingerprint):
            combined_feature_vector.append(x.tolist() + y.tolist() + z.tolist())
        combined_feature_vector = np.asarray(combined_feature_vector)

        all_feature_vectors.append(combined_feature_vector)

    # Default Classifier
    clf = XGBClassifier(random_state=1, n_jobs=-1)

    clf.fit(all_feature_vectors[0], activity)
    importances = clf.feature_importances_
    boolean_mask = importances != 0

    if molecules_2 is None:
        new_feature_vector = all_feature_vectors[0][:, boolean_mask]
    else:
        new_feature_vector = all_feature_vectors[1][:, boolean_mask]

    return new_feature_vector


def select_most_important_features(classifier, feature_vector_basis, number_features, activity, vector_to_reduce):
    """
    Creates a feature vector using the k most important features

    :param classifier: the classifier object to chose the best features from
    :param feature_vector_basis: feature vectors for training and chose the most important features
    :param number_features: number of best (most important) features for the final feature vector
    :param activity: list of class labels of the feature_vector_basis
    :param vector_to_reduce: feature vectors on which the k most important feature selection is applied on
    :return:
    """
    classifier.fit(feature_vector_basis, activity)
    return vector_to_reduce[:, classifier.feature_importances_.argsort()[::-1][:number_features]]


def calculate_scores(y_true, y_pred):
    """
    Calculates and prints scores

    :param y_true: actual class labels
    :param y_pred: predicted class labels
    :return: None
    """
    print("MCC:\t" + str(matthews_corrcoef(y_true, y_pred)))
    print("ACC:\t" + str(accuracy_score(y_true, y_pred)))
    print("SE:\t" + str(recall_score(y_true, y_pred)))
    print("SP:\t" + str(my_specificity_score(y_true, y_pred)))
    print("ROCAUC:\t" + str(roc_auc_score(y_true, y_pred)))


def my_specificity_score(labels, predictions):
    """
    specificity scoring function

    :param labels: actual class labels
    :param predictions: predicted class labels
    :return: specificity score
    """
    tp = fp = tn = fn = 0
    for x in range(len(labels)):
        if (predictions[x] == labels[x]) and (labels[x] == 1):
            tp += 1
        elif (predictions[x] != labels[x]) and (labels[x] == 1):
            fn += 1
        elif (predictions[x] == labels[x]) and (labels[x] == 0):
            tn += 1
        elif (predictions[x] != labels[x]) and (labels[x] == 0):
            fp += 1
    score = tn / (tn + fp)
    return score


def write_output(orig_smiles_predict, labels_predict, labels_original, my_output):
    """
    write output tab separated:
    SMILES_String \t true_class_label \t predicted_class_label \n

    :param orig_smiles_predict: smiles strings
    :param labels_predict: prediceted labels
    :param labels_original: actual class labels
    :param my_output: output path
    :return: None
    """
    with open(my_output, 'w+') as f:
        for smiles, label, label_orig in zip(orig_smiles_predict, labels_predict, labels_original):
            if label == 0:
                label = -1
            if label_orig == 0:
                label_orig = -1
            f.write('\t'.join([smiles, str(label_orig), str(label)]) + '\n')
