import os
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from cp_posh.analysis.similarity import get_embedding_cosine_similarity


def get_stringdb_data(
    selected_genes: List[str], physical_db: bool = False
) -> pd.DataFrame:
    """
    load gene-gene interaction data from stringdb and subset to selected genes

    Parameters
    ----------
    selected_genes : List[str]
        list of genes to subset
    physical_db : bool
        whether to use physical interactions only, by default False

    Returns
    -------
    pd.DataFrame
        dataframe of stringDB relations of shape (len(selected_genes), len(selected_genes))
        The value at (i, j) is the interaction score between gene i and gene j
    """
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    if physical_db:
        stringdb_links = pd.read_csv(
            os.path.join(
                curr_dir, "stringdb_data", "9606.protein.physical.links.v11.5.txt"
            ),
            delimiter=" ",
        )
    else:
        stringdb_links = pd.read_csv(
            os.path.join(curr_dir, "stringdb_data", "9606.protein.links.v11.5.txt"),
            delimiter=" ",
        )
    stringdb_names = pd.read_csv(
        os.path.join(curr_dir, "stringdb_data", "9606.protein.info.v11.5.txt"),
        delimiter="\t",
    )
    name_map = stringdb_names.set_index("#string_protein_id").to_dict()[
        "preferred_name"
    ]
    stringdb_links["protein1"] = stringdb_links["protein1"].apply(lambda x: name_map[x])
    stringdb_links["protein2"] = stringdb_links["protein2"].apply(lambda x: name_map[x])

    # normalize score to be between 0 and 1
    stringdb_links["norm_score"] = stringdb_links["combined_score"] / 1000

    stringdb_links_gene_subset = stringdb_links[
        stringdb_links.protein1.isin(selected_genes)
    ]
    stringdb_links_gene_subset = stringdb_links_gene_subset[
        stringdb_links_gene_subset.protein2.isin(selected_genes)
    ]
    # pivot table to get matrix of gene-gene interactions
    stringdb_matrix = pd.pivot_table(
        stringdb_links_gene_subset,
        index=["protein1"],
        columns=["protein2"],
        values="norm_score",
    )
    return stringdb_matrix


def compute_aggregate_feature_similarity_matrix(
    aggregate_features: pd.DataFrame,
    select_genes: List[str],
    gene_id_column: str = "gene_id",
    abs_value: bool = True,
) -> pd.DataFrame:
    """
    Compute cosine similarity matrix of aggregate features for a select subset of genes

    Parameters
    ----------
    aggregate_features : pd.DataFrame
        input aggregate features
    select_genes : List[str]
        selected genes to compute similarity matrix
    gene_id_column: str
        column name that contains the gene IDs,
        by default "gene_id"
    abs_value: bool
        whether to return the absolute value of the similarity matrix

    Returns
    -------
    pd.DataFrame
        dataframe of aggregate feature similarities
    """
    aggregate_features = aggregate_features[
        aggregate_features.index.get_level_values(gene_id_column).isin(select_genes)
    ]

    similarity = get_embedding_cosine_similarity(aggregate_features)

    if abs_value:
        similarity = similarity.abs()

    return similarity


def compute_stringdb_roc_prc_metrics(
    aggregate_features: pd.DataFrame,
    select_genes: List[str],
    stringdb_threshold: float = 0.95,
    gene_id_column: str = "gene_id",
    abs_value: bool = False,
    stringdb_matrix: Optional[pd.DataFrame] = None,
) -> Dict[str, Union[int, float, np.ndarray]]:
    """
    Metric as defined in "A Pooled Cell Painting CRISPR Screening Platform Enables
    de novo Inference of Gene Function by Self-supervised Deep Learning"
    https://www.biorxiv.org/content/10.1101/2023.08.13.553051v1
    Compute ROC and Precision-Recall metrics for the aggregate feature similarity matrix

    Parameters
    ----------
    aggregate_features : pd.DataFrame
        input aggregate features
    select_genes : list, optional
        selected genes to compute similarity matrix, by default []
    stringdb_threshold : float, optional
        stringdb threshold, by default 0.95
    gene_id_column: str
        column name that contains the gene IDs
    abs_value: Optional[bool]
        whether to return the absolute value of the similarity matrix
    stringdb_matrix : Optional[pd.DataFrame]
        stringdb matrix obtained from `get_stringdb_data`, default is None

    Returns
    -------
    Dict[str, Union[int, float, np.ndarray]]
        dictionary of ROC metrics
    """
    matrix = compute_aggregate_feature_similarity_matrix(
        aggregate_features,
        select_genes,
        gene_id_column=gene_id_column,
        abs_value=abs_value,
    )

    if stringdb_matrix is None:
        stringdb_matrix = get_stringdb_data(select_genes)

    metrics = {}
    stringdb_scores = []
    feature_matrix_scores = []

    genes = np.array(select_genes)
    genes = genes[(genes != "nontargeting") & (genes != "intergenic")]

    for g1 in range(len(genes)):
        gene1 = genes[g1]
        for g2 in range(g1 + 1, len(genes)):
            gene2 = genes[g2]
            if gene1 != gene2:
                if (
                    gene1 in stringdb_matrix.index
                    and gene2 in stringdb_matrix.index
                    and gene1 in matrix.index
                    and gene2 in matrix.index
                ):
                    sdb_val = stringdb_matrix.loc[gene1, gene2]
                    if not np.isnan(sdb_val):
                        stringdb_scores.append(sdb_val)
                        feature_matrix_scores.append(matrix.loc[gene1, gene2])
                    else:
                        stringdb_scores.append(0)
                        feature_matrix_scores.append(matrix.loc[gene1, gene2])

    sim_string = np.array(stringdb_scores)
    sim_features = np.array(feature_matrix_scores)
    string_thr = stringdb_threshold
    y_true = sim_string[(sim_string > string_thr) | (sim_string == 0)].copy()
    y_true[y_true > string_thr] = 1
    y_true = y_true.astype(int)

    y_score = np.array(
        sim_features[(sim_string > string_thr) | (sim_string == 0)].copy()
    )
    fprs, tprs, thrs = roc_curve(y_true, y_score)
    interp_tpr_5 = np.interp(0.05, fprs, tprs)
    interp_tpr_10 = np.interp(0.1, fprs, tprs)

    precisions, recalls, thrs_prcs = precision_recall_curve(y_true, y_score)
    mavp = average_precision_score(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    metrics["roc_tprs"] = tprs
    metrics["roc_fprs"] = fprs
    metrics["roc_thresholds"] = thrs
    metrics["roc_tpr_at_5pc_fpr"] = interp_tpr_5
    metrics["roc_tpr_at_10pc_fpr"] = interp_tpr_10
    metrics["roc_auc"] = auc

    metrics["prc_precisions"] = precisions
    metrics["prc_recalls"] = recalls
    metrics["prc_thresholds"] = thrs_prcs
    metrics["average_precision_score"] = mavp

    return metrics
