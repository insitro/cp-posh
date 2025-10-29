from typing import List, Optional
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import pairwise_distances


def get_embedding_cosine_similarity(embeddings: pd.DataFrame) -> pd.DataFrame:
    """
    compute embedding similarity as cosine similarity between embeddings

    Parameters
    ----------
    embeddings : pd.DataFrame
        input embeddings

    Returns
    -------
    pd.DataFrame
        embedding similarity
    """
    similarity = 1 - pd.DataFrame(
        pairwise_distances(embeddings, metric="cosine"),
        index=embeddings.index,
        columns=embeddings.index,
    )
    return similarity


def get_gRNA_scores(
    embeddings: pd.DataFrame,
    select_genes: Optional[List[str]] = None,
    plotstyle: Optional[str] = "box",
    path_to_save: Optional[str] = None,
) -> pd.DataFrame:
    """ ""
    Determine the cosine similarities of gRNAs targeting the same gene

    Parameters
    ----------
    embeddings : pd.DataFrame
        input embeddings.  Must be a multiIndex dataframe containing gene_id and barcode columns
    select_genes : Optional[List[str]]
        a subset list of genes to calculate.  If none, will calculate similarites of all genes
    plotstyle : Optional[str]
        'box' or 'dot', determining the format of the plot to create
    path_to_save: Optional[str]
        Name of the PDF or SVG which will be automatically saved upon completion. Recommended to end with '.pdf' or '.svg'

    Returns
    -------
    pd.DataFrame
        gRNA scores
    """

    if select_genes is not None:
        embeddings_orig = embeddings.copy(deep=True)[
            embeddings.index.get_level_values("gene_id").isin(select_genes)
        ]
    else:
        embeddings_orig = embeddings.copy(deep=True)

    embeddings_simcalc = embeddings_orig.copy(deep=True)
    embeddings_simcalc.index = embeddings_orig.index.get_level_values("barcode")
    guide_similarity_mat = get_embedding_cosine_similarity(embeddings_simcalc)
    guide_similarity_mat.index = embeddings_orig.index
    gRNAscores = pd.DataFrame(index=guide_similarity_mat.index)
    gRNAscores["score"] = "NA"

    for gRNA in gRNAscores.index:
        gRNAset = guide_similarity_mat[
            guide_similarity_mat.index.get_level_values("gene_id") == gRNA[1]
        ].index
        gRNAscore = -1.0
        for matchgRNA in gRNAset:
            gRNAscore = gRNAscore + guide_similarity_mat.loc[gRNA, matchgRNA[0]]
        gRNAscore = gRNAscore / (len(gRNAset) - 1)
        gRNAscores.loc[gRNA].score = gRNAscore

    if plotstyle == "box":
        _ = gRNAscores.boxplot(figsize=(26, 8), column=["score"], by="gene_id")
        plt.xticks(rotation=90)
        plt.ylim(-1, 1)
        plt.axhline(
            (guide_similarity_mat.sum().sum() - len(guide_similarity_mat))
            / (len(guide_similarity_mat) ** 2),
            color="red",
            linestyle="-",
        )

    if plotstyle == "dot":
        scatterplot = gRNAscores.copy(deep=True)
        scatterplot.index = gRNAscores.index.get_level_values("gene_id")
        scatterplot.sort_index().reset_index().plot.scatter(
            x="gene_id", y="score", figsize=(26, 8), alpha=0.5, c="k"
        )
        plt.xticks(rotation=90)
        plt.ylim(-1, 1)
        plt.axhline(
            (guide_similarity_mat.sum().sum() - len(guide_similarity_mat))
            / (len(guide_similarity_mat) ** 2),
            color="red",
            linestyle="-",
        )
        plt.margins(x=0.003)

    if path_to_save is not None:
        plt.savefig(path_to_save, dpi=300, bbox_inches="tight")
    plt.show()

    return gRNAscores
