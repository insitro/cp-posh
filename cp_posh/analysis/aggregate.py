from typing import List
import pandas as pd


def _aggregate_embeddings(
    embeddings: pd.DataFrame,
    method: str = "mean",
    group_columns: List[str] = ["barcode", "gene_id"],
) -> pd.DataFrame:
    """
    Aggregate embeddings by taking the mean or median of the embeddings for
    each group of rows.

    Parameters
    ----------
    embeddings_input : pd.DataFrame
        input embeddings
    method : str, optional
        aggregation method, by default "mean"
    group_columns : list, optional
        columns to group by, by default ['barcode', 'gene_id']

    Returns
    -------
    pd.DataFrame
        aggregate embeddings
    """

    embeddings_cp = embeddings.copy(deep=True)

    if method == "mean":
        agg_embeddings = embeddings_cp.groupby(group_columns).mean()
    elif method == "median":
        agg_embeddings = embeddings_cp.groupby(group_columns).median()

    return agg_embeddings


def aggregate_embeddings_by_gRNA(
    embeddings: pd.DataFrame, method: str = "mean"
) -> pd.DataFrame:
    """
    Aggregate embeddings by taking the mean or median of the embeddings for
    each guide.

    Parameters
    ----------
    embeddings : pd.DataFrame
        input embeddings
    method : str, optional
        aggregation method, by default "mean"

    Returns
    -------
    pd.DataFrame
        aggregate embeddings
    """
    return _aggregate_embeddings(
        embeddings, method, group_columns=["barcode", "gene_id"]
    )


def aggregate_embeddings_by_gene(
    embeddings: pd.DataFrame, method: str = "median"
) -> pd.DataFrame:
    """
    Aggregate embeddings by taking the mean or median of the embeddings for
    each gene.

    Parameters
    ----------
    embeddings : pd.DataFrame
        input embeddings
    method : str, optional
        aggregation method, by default "median"

    Returns
    -------
    pd.DataFrame
        aggregate embeddings
    """
    return _aggregate_embeddings(embeddings, method, group_columns=["gene_id"])
