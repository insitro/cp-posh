import pickle
from typing import Any, Dict, List, Sequence, cast

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
import argparse


def _train_classifier_gene(train_X: np.ndarray, train_y: np.ndarray) -> Pipeline:
    """
    train binary classifier between control and gene KO

    Parameters
    ----------
    train_X : np.ndarray
        training data of shape (n_samples, n_features)
    train_y : _type_
        binary training labels of shape (n_samples, 1)

    Returns
    -------
    Pipeline
        trained classifier pipeline
    """
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            Cs=5,
            n_jobs=16,
            random_state=42,
            max_iter=300,
            solver="saga",
            penalty="elasticnet",
            l1_ratios=[0.5],
        ),
    )
    clf = clf.fit(train_X, train_y)
    return clf


def train_validate_per_split(
    embeddings: pd.DataFrame,
    validation_plate_well: str,
    control_gene_id: str = "intergenic",
) -> Sequence[Any]:
    """
    train and validate linear model for a single split

    Parameters
    ----------
    embeddings : pd.DataFrame
        embeddings dataframe
    validation_plate_well : str
        plate well to hold out for validation
    control_gene_id : str, optional
        control gene ID, by default "intergenic"

    Returns
    -------
    Tuple[Any]
        tuple of evaluation metrics
    """
    np.random.seed(42)
    scores_sampled: Dict[str, List[Any]] = {
        g: [] for g in embeddings.index.get_level_values("gene_id").unique()
    }
    aurocs_sampled: Dict[str, List[Any]] = {
        g: [] for g in embeddings.index.get_level_values("gene_id").unique()
    }
    auprcs_sampled: Dict[str, List[Any]] = {
        g: [] for g in embeddings.index.get_level_values("gene_id").unique()
    }

    scores_all: Dict[str, List[Any]] = {
        g: [] for g in embeddings.index.get_level_values("gene_id").unique()
    }
    aurocs_all: Dict[str, List[Any]] = {
        g: [] for g in embeddings.index.get_level_values("gene_id").unique()
    }
    auprcs_all: Dict[str, List[Any]] = {
        g: [] for g in embeddings.index.get_level_values("gene_id").unique()
    }

    random_ap: Dict[str, List[Any]] = {
        g: [] for g in embeddings.index.get_level_values("gene_id").unique()
    }

    models: Dict[str, List[Any]] = {
        g: [] for g in embeddings.index.get_level_values("gene_id").unique()
    }
    train_sizes: Dict[str, List[Any]] = {
        g: [] for g in embeddings.index.get_level_values("gene_id").unique()
    }
    test_sizes: Dict[str, List[Any]] = {
        g: [] for g in embeddings.index.get_level_values("gene_id").unique()
    }

    pred_scores = []

    train = embeddings[
        embeddings.index.get_level_values("plate_well") != validation_plate_well
    ]
    train_X = train.values
    train_Y = train.index.get_level_values("gene_id").values

    test_sampled = embeddings[
        embeddings.index.get_level_values("plate_well") == validation_plate_well
    ]
    test_X = test_sampled.values
    test_Y = test_sampled.index.get_level_values("gene_id").values

    test_all = embeddings[
        embeddings.index.get_level_values("plate_well") == validation_plate_well
    ]
    test_X_all = test_all.values
    test_Y_all = test_all.index.get_level_values("gene_id").values
    clf_refs = []
    gene_ids = []

    for gene_id in np.unique(test_Y):
        if gene_id == control_gene_id:
            continue

        train_x_pos = train_X[np.where(train_Y == gene_id)]
        train_x_neg = train_X[np.where(train_Y == control_gene_id)]

        if len(train_x_pos) > len(train_x_neg):
            train_x_pos = train_x_pos[
                np.random.choice(
                    np.arange(len(train_x_pos)), len(train_x_neg), replace=False
                )
            ]
        else:
            train_x_neg = train_x_neg[
                np.random.choice(
                    np.arange(len(train_x_neg)), len(train_x_pos), replace=False
                )
            ]

        train_X_bin = np.concatenate([train_x_pos, train_x_neg], axis=0)
        train_y_bin = np.array([1] * len(train_x_pos) + [0] * len(train_x_neg))

        clf_refs.append(delayed(_train_classifier_gene)(train_X_bin, train_y_bin))
        gene_ids.append(gene_id)
        train_sizes[gene_id].append(len(train_X_bin))

    print("Running Linear Evaluations")
    clfs = Parallel(n_jobs=16)(clf_refs)
    clfs = [clf for clf in clfs]

    for gene_id in gene_ids:
        clf: Pipeline = cast(Pipeline, clfs[gene_ids.index(gene_id)])

        test_x_pos = test_X[np.where(test_Y == gene_id)]
        test_x_neg = test_X[np.where(test_Y == control_gene_id)]

        if len(test_x_pos) > len(test_x_neg):
            test_x_pos = test_x_pos[
                np.random.choice(
                    np.arange(len(test_x_pos)), len(test_x_neg), replace=False
                )
            ]
        else:
            test_x_neg = test_x_neg[
                np.random.choice(
                    np.arange(len(test_x_neg)), len(test_x_pos), replace=False
                )
            ]

        test_X_bin = np.concatenate([test_x_pos, test_x_neg], axis=0)
        test_y_bin = np.array([1] * len(test_x_pos) + [0] * len(test_x_neg))

        test_X_bin_all = np.concatenate(
            [
                test_X_all[np.where(test_Y_all == gene_id)],
                test_X_all[np.where(test_Y_all == control_gene_id)],
            ],
            axis=0,
        )
        test_y_bin_all = np.array(
            [1] * len(test_X_all[np.where(test_Y_all == gene_id)])
            + [0] * len(test_X_all[np.where(test_Y_all == control_gene_id)])
        )

        predicted_probs_sampled = clf.predict_proba(test_X_bin)
        auroc_sampled = roc_auc_score(test_y_bin, predicted_probs_sampled[:, 1])
        auprc_sampled = average_precision_score(
            test_y_bin, predicted_probs_sampled[:, 1]
        )

        preds_sampled = clf.predict(test_X_bin)
        accuracy_sampled = accuracy_score(test_y_bin, preds_sampled)

        predicted_probs_all = clf.predict_proba(test_X_bin_all)
        auroc_all = roc_auc_score(test_y_bin_all, predicted_probs_all[:, 1])
        auprc_all = average_precision_score(test_y_bin_all, predicted_probs_all[:, 1])

        preds_all = clf.predict(test_X_bin_all)
        accuracy_all = accuracy_score(test_y_bin_all, preds_all)

        # Generate random predictions
        random_scores = np.random.rand(len(test_y_bin_all))

        # Compute the AP for these random predictions
        random_chance_ap = average_precision_score(test_y_bin_all, random_scores)

        gene_group_data = embeddings[
            (embeddings.index.get_level_values("plate_well") == validation_plate_well)
            & (embeddings.index.get_level_values("gene_id") == gene_id)
        ]
        pred_score = pd.DataFrame(
            {
                "ID": gene_group_data.index.get_level_values("ID"),
                "predicted_phenotype_probability": clf.predict_proba(
                    gene_group_data.values
                )[:, 1],
            }
        )
        scores_sampled[gene_id].append(accuracy_sampled)
        aurocs_sampled[gene_id].append(auroc_sampled)
        auprcs_sampled[gene_id].append(auprc_sampled)

        scores_all[gene_id].append(accuracy_all)
        aurocs_all[gene_id].append(auroc_all)
        auprcs_all[gene_id].append(auprc_all)
        random_ap[gene_id].append(random_chance_ap)
        test_sizes[gene_id].append(len(test_X_bin))

        models[gene_id].append(clf)
        pred_scores.append(pred_score)
        print(
            gene_id,
            accuracy_sampled,
            auroc_sampled,
            auprc_sampled,
            accuracy_all,
            auroc_all,
            auprc_all,
            random_chance_ap,
            train_sizes[gene_id],
            test_sizes[gene_id],
        )

    return (
        scores_sampled,
        aurocs_sampled,
        auprcs_sampled,
        scores_all,
        aurocs_all,
        auprcs_all,
        random_ap,
        models,
        pd.concat(pred_scores),
    )


def train_linear_model_and_evaluate_leave_well_out(
    embeddings: pd.DataFrame, control_gene_id: str = "intergenic"
) -> Dict[str, Any]:
    """
    training and validation of linear models in a leave well out manner
    across wells

    Parameters
    ----------
    embeddings : pd.DataFrame
        embeddings dataframe
    control_gene_id : str, optional
        control gene ID, by default "intergenic"

    Returns
    -------
    Dict[str, Any]
        evaluation metrics
    """
    scores_sampled: Dict[str, List[Any]] = {
        g: [] for g in embeddings.index.get_level_values("gene_id").unique()
    }
    aurocs_sampled: Dict[str, List[Any]] = {
        g: [] for g in embeddings.index.get_level_values("gene_id").unique()
    }
    auprcs_sampled: Dict[str, List[Any]] = {
        g: [] for g in embeddings.index.get_level_values("gene_id").unique()
    }
    models: Dict[str, List[Any]] = {
        g: [] for g in embeddings.index.get_level_values("gene_id").unique()
    }

    scores_all: Dict[str, List[Any]] = {
        g: [] for g in embeddings.index.get_level_values("gene_id").unique()
    }
    aurocs_all: Dict[str, List[Any]] = {
        g: [] for g in embeddings.index.get_level_values("gene_id").unique()
    }
    auprcs_all: Dict[str, List[Any]] = {
        g: [] for g in embeddings.index.get_level_values("gene_id").unique()
    }
    random_aps: Dict[str, List[Any]] = {
        g: [] for g in embeddings.index.get_level_values("gene_id").unique()
    }

    pred_scores = []

    outputs = []
    for validation_plate_well in embeddings.index.get_level_values(
        "plate_well"
    ).unique():
        print(validation_plate_well)
        outputs.append(
            train_validate_per_split(
                embeddings,
                validation_plate_well,
                control_gene_id=control_gene_id,
            )
        )

    for output in outputs:
        for gene_id in output[0].keys():
            scores_sampled[gene_id].extend(output[0][gene_id])
            aurocs_sampled[gene_id].extend(output[1][gene_id])
            auprcs_sampled[gene_id].extend(output[2][gene_id])

            scores_all[gene_id].extend(output[3][gene_id])
            aurocs_all[gene_id].extend(output[4][gene_id])
            auprcs_all[gene_id].extend(output[5][gene_id])

            random_aps[gene_id].extend(output[6][gene_id])

            models[gene_id].extend(output[7][gene_id])
        pred_scores.append(output[8])

    return {
        "accuracy_scores_sampled": scores_sampled,
        "aurocs_sampled": aurocs_sampled,
        "auprcs_sampled": auprcs_sampled,
        "accuracy_scores_all": scores_all,
        "aurocs_all": aurocs_all,
        "auprcs_all": auprcs_all,
        "random_average_precision": random_aps,
        "models": models,
        "xval_prediction_scores": pd.concat(pred_scores),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train and evaluate linear models on embeddings."
    )
    parser.add_argument(
        "--embedding_path", type=str, help="path to embedding parquet file"
    )
    parser.add_argument(
        "--control_gene_id",
        type=str,
        default="intergenic",
        help="control gene ID used as null treatment for linear classification, default is 'intergenic'.",
    )
    parser.add_argument(
        "--output_path", type=str, help="path to save evaluation results"
    )
    args = parser.parse_args()

    # read embedding parquet file
    embeddings = pd.read_parquet(args.embedding_path)

    # cross validation metrics for the logstic regression model
    results = train_linear_model_and_evaluate_leave_well_out(
        embeddings, control_gene_id=args.control_gene_id
    )

    # save results
    pickle.dump(results, open(args.output_path, "wb"))
