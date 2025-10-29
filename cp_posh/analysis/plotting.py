import pandas as pd
import scipy
from scipy import stats
from sklearn.neighbors import radius_neighbors_graph
import umap
import numpy as np
import leidenalg as la
import igraph as ig
import matplotlib.pyplot as plt
from adjustText import adjust_text
import seaborn as sns
from typing import List, Optional, Tuple, Dict, Any
from cp_posh.analysis import stringdb
import statsmodels.stats.multitest as multi
import matplotlib as mpl

# beautify plots
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["font.size"] = 18
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.linewidth"] = 2


def scatter_plot(
    xkey: str,
    ykey: str,
    metrics: pd.DataFrame,
    metrics_null: pd.DataFrame = None,
    annotation_threshold: float = 0.75,
    null_cmap: str = "tab:orange",
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    path_to_save: Optional[str] = None,
) -> None:
    """
    Scatter plot of two metrics with null distribution

    Parameters
    ----------
    xkey : str
        x-axis key column in metrics and metrics_null
    ykey : str
        y-axis key column in metrics and metrics_null
    metrics : pd.DataFrame
        metrics dataframe
    metrics_null : Optional[pd.DataFrame], optional
        null distribution dataframe
    annotation_threshold : float, optional
        threshold for annotation, by default 0.75
    null_cmap : _type_, optional
        coloir map for null distribution, by default 'tab:orange'
    xlabel : str, optional
        xlabel, by default ""
    ylabel : str, optional
        ylabel, by default ""
    title : str, optional
        title, by default ""
    path_to_save : Optional[str], optional
        path to save the plot, by default None
    """
    plt.figure()
    plt.scatter(
        metrics[xkey],
        metrics[ykey],
        marker="o",
        s=20,
        alpha=0.5,
        facecolor="None",
        color="black",
    )
    if metrics_null is not None:
        plt.scatter(
            metrics_null[xkey],
            metrics_null[ykey],
            marker=".",
            s=20,
            alpha=0.5,
            facecolor="None",
            color=null_cmap,
        )

    plt.plot(np.arange(0.5, 0.95, 0.1), np.arange(0.5, 0.95, 0.1), "g--")
    # Add text annotations for labels using adjustText
    rows_to_show_annotations = metrics[
        (metrics[xkey] > annotation_threshold) | (metrics[ykey] > annotation_threshold)
    ]
    annotations = [
        plt.text(
            rows_to_show_annotations[xkey][i],
            rows_to_show_annotations[ykey][i],
            label,
            ha="center",
            va="center",
            fontsize=12,
        )
        for i, label in enumerate(rows_to_show_annotations.index.values)
    ]
    adjust_text(annotations, arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=15)
    if path_to_save is not None:
        plt.savefig(path_to_save, dpi=300, bbox_inches="tight")
    plt.show()


def barplot(
    x_values: np.ndarray,
    y_values: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    ypad: float,
    tp: str = "int",
    figsize: Tuple[int, int] = (5, 2),
    path_to_save: Optional[str] = None,
) -> None:
    """
    Create a horizontal bar plot

    Parameters
    ----------
    x_values : np.ndarray
        x-axis values (labels)
    y_values : np.ndarray
        y-axis values (bar heights)
    title : str
        title of the plot
    xlabel : str
        x-axis label
    ylabel : str
        y-axis label
    ypad : float
        y-axis padding for annotations
    tp : str, optional
        type of y-values, by default 'int'
    figsize : Tuple[int, int], optional
        figure size, by default (5, 2)
    path_to_save : Optional[str], optional
        path to save the plot, by default None
    """
    colors = sns.color_palette("BuGn", len(y_values))

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(x_values, y_values, color=colors, height=0.5, edgecolor="black")

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=12)

    ax.tick_params(
        axis="both", which="major", labelsize=12, width=2, length=6, direction="in"
    )
    ax.tick_params(axis="both", which="minor", width=1, length=4, direction="in")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(top=False, right=False)
    ax.set_yticklabels(x_values, ha="right", rotation_mode="anchor")

    for i, y in enumerate(y_values):
        if tp == "int":
            v = f"{int(y)}"
        else:
            v = f"{y:0.2f}"
        ax.text(y + ypad, i - 0.05, v, color="black", fontsize=15)

    ax.set_title(title, fontsize=12)

    if path_to_save is not None:
        plt.savefig(path_to_save, dpi=300, bbox_inches="tight")

    plt.show()


def compute_plot_umap(
    gene_embeddings: pd.DataFrame,
    selected_genes: List[str],
    within_gene_gRNA_similarity_scores: Optional[pd.DataFrame] = None,
    neighbors: int = 10,
    leiden_resolution: float = 0.2,
    max_community_size: int = 20,
    adjust_texts: bool = False,
    clusters: Optional[Dict[str, int]] = None,
    dot_size_min: Optional[int] = 20,
    dot_size_max: Optional[int] = 750,
    font_size: Optional[int] = 12,
    random_state: int = 42,
    path_to_save: Optional[str] = None,
) -> umap.UMAP:
    """
    Compute and plot UMAP for gene embeddings

    Parameters
    ----------
    gene_embeddings : pd.DataFrame
        gene embeddings dataframe
    selected_genes : List[str]
        selected genes to plot
    within_gene_gRNA_similarity_scores : Optional[pd.DataFrame], optional
        within gene gRNA similarity scores, by default None
    neighbors : int, optional
        number of neighbors, by default 10
    leiden_resolution : float, optional
        leiden resolution, by default 0.2
    max_community_size : int, optional
        max community size, by default 20
    adjust_texts : bool, optional
        adjust text annotations, by default False
    clusters : Optional[Dict[str, int]], optional
        cluster labels, by default None
    dot_size_min : Optional[int], optional
        min dot size, by default 20
    dot_size_max : Optional[int], optional
        max dot size, by default 750
    font_size : Optional[int], optional
        font size for gene name annotations, by default 12
    path_to_save : Optional[str], optional
        path to save the plot, by default None

    Returns
    -------
    umap.UMAP
        UMAP reducer object
    """

    reducer = umap.UMAP(
        random_state=random_state, n_neighbors=neighbors, metric="cosine"
    )
    gene_embeddings = gene_embeddings[
        gene_embeddings.index.get_level_values("gene_id").isin(selected_genes)
    ]
    umaps: np.ndarray = reducer.fit_transform(gene_embeddings)

    # Create a DataFrame with umap results
    df = pd.DataFrame(
        {
            "x": umaps[:, 0],
            "y": umaps[:, 1],
            "labels": gene_embeddings.index.get_level_values("gene_id").values,
        }
    )

    if within_gene_gRNA_similarity_scores is not None:
        gene_scores = (
            within_gene_gRNA_similarity_scores.groupby("gene_id").median().reset_index()
        )
        df = df.merge(gene_scores, left_on="labels", right_on="gene_id")

    if clusters is None:
        adjacency = radius_neighbors_graph(
            df[["x", "y"]].values,
            radius=0.5,
            metric="euclidean",
            mode="connectivity",
            include_self=False,
        )

        # Create a graph object from the adjacency matrix
        graph = ig.Graph.Adjacency(adjacency.toarray().tolist())
        partition = la.find_partition(
            graph,
            la.RBERVertexPartition,
            resolution_parameter=leiden_resolution,
            max_comm_size=max_community_size,
        )

        # Assign labels to clusters
        clustering_labels = partition.membership
    else:
        clustering_labels = [clusters[i] for i in df.labels.values]

    # Create the scatter plot
    plt.figure(figsize=(20, 20))

    size = np.maximum(np.array(df.score.values) * 0, dot_size_min)  # type: ignore
    if within_gene_gRNA_similarity_scores is not None:
        size = np.maximum(np.array(df.score.values) * dot_size_max, dot_size_min)  # type: ignore

    plt.scatter(
        np.array(df.x.values),
        np.array(df.y.values),
        c=clustering_labels,
        cmap="rainbow",
        s=list(size),
        alpha=0.8,
        edgecolors="k",
        linewidths=1,
    )

    # Add text annotations for labels using adjustText
    annotations = [
        plt.text(df.x[i], df.y[i], label, ha="center", va="center", fontsize=font_size)
        for i, label in enumerate(df.labels.values)
    ]
    if adjust_texts:
        print("adjust_text = TRUE, may slow down UMAP plotting time...")
        adjust_text(
            annotations, arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5)
        )

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Remove the top and right spines
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # Customize the plot
    plt.xlabel("UMAP 1", fontsize=30)
    plt.ylabel("UMAP 2", fontsize=30)

    # Set the plot quality to high resolution
    if path_to_save is not None:
        plt.savefig(path_to_save, dpi=300, bbox_inches="tight")
    plt.show()

    return reducer


def plot_stringdb_violin(
    gene_similarity: pd.DataFrame,
    stringdb_threshold_lower: float = 0.05,
    stringdb_threshold_upper: float = 0.95,
    path_to_save: Optional[str] = None,
):
    """
    Plots a violin plot of a calculated gene-gene similarity matrix after binning on low, medium, and high predicted connection using an established protein-protein interaction network database (StringDB).

    Parameters
    ----------
    gene_similarity: NumPy Array
        An array of similarities across the genes.  Columns and indices are GeneIDs. Please ensure that these GeneIDs match the format used for StringDB lookup! Attempts will be made to correct for slight mismatches/synonyms.
        Run gene_similarity = pd.read_csv('correlations.csv',index_col = 0) if an example dataset is needed.
    stringdb_threshold_lower: scalar between 0 and stringdb_threshold_upper
        The cutoff between "low" and "mid" strength bins for stringDB edges
    stringdb_threshold_upper: scalar between stringdb_threshold_lower and 1
        The cutoff between "mid" and "high" strength bins for stringDB edges
    plot_savefile: string
        Filename for the plot to be saved to. (pdf or svg recommended)


    Outputs
    -------
    (LH, LM, MH) =
    KS 2-sample 2-tail test between LOW and HIGH bin distributions
    KS 2-sample 2-tail test between LOW and MID bin distributions
    KS 2-sample 2-tail test between MID and HIGH bin distributions
    """

    print("Using genes from correlation matrix, pulling stringdb network...")

    # Catch if multiindexed dataframe is fed into function
    if gene_similarity.index.nlevels > 1:
        gene_similarity.index = gene_similarity.index.get_level_values("gene_id")
        gene_similarity.columns = gene_similarity.columns.get_level_values("gene_id")

    gene_similarity = gene_similarity.drop(
        index=gene_similarity.filter(like="intergenic").columns
    )
    gene_similarity = gene_similarity.drop(
        index=gene_similarity.filter(like="nontargeting").columns
    )

    gene_similarity = gene_similarity.drop(
        columns=gene_similarity.filter(like="intergenic").columns
    )
    gene_similarity = gene_similarity.drop(
        columns=gene_similarity.filter(like="nontargeting").columns
    )

    np.fill_diagonal(gene_similarity.values, 0)
    genes = gene_similarity.index.values

    sim_string = stringdb.get_stringdb_data(genes)
    sim_morph = gene_similarity
    m, n = sim_string.shape
    sim_string[sim_string.isna()] = 0
    sim_string[:] = np.where(np.arange(m)[:, None] >= np.arange(n), -100, sim_string)

    stringmatched_values = (
        sim_morph[sim_string >= stringdb_threshold_upper].stack().values
    )
    stringmid_values = (
        sim_morph[
            np.logical_and(
                sim_string >= stringdb_threshold_lower,
                sim_string < stringdb_threshold_upper,
            )
        ]
        .stack()
        .values
    )
    stringunmatched_values = (
        sim_morph[
            np.logical_and(sim_string > -99, sim_string < stringdb_threshold_lower)
        ]
        .stack()
        .values
    )

    fig, ax2 = plt.subplots()
    fig.set_size_inches(8, 6)

    data = [stringunmatched_values, stringmid_values, stringmatched_values]

    ax2.set_title("Morphological Edge Weights vs. StringDB Edges")
    parts = ax2.violinplot(data, showmeans=False, showmedians=False, showextrema=False)

    for pc in parts["bodies"]:
        pc.set_facecolor("blue")
        pc.set_edgecolor("black")
        pc.set_alpha(1)

    matched_min, matched_q1, matched_median, matched_q3, matched_max = np.percentile(
        stringmatched_values, [0, 25, 50, 75, 100]
    )
    midmatched_min, midmatched_q1, midmatched_median, midmatched_q3, midmatched_max = (
        np.percentile(stringmid_values, [0, 25, 50, 75, 100])
    )
    unmatched_min, unmatched_q1, unmatched_median, unmatched_q3, unmatched_max = (
        np.percentile(stringunmatched_values, [0, 25, 50, 75, 100])
    )

    ax2.scatter(3, matched_median, marker="o", color="white", s=30, zorder=3)
    ax2.scatter(2, midmatched_median, marker="o", color="white", s=30, zorder=3)
    ax2.scatter(1, unmatched_median, marker="o", color="white", s=30, zorder=3)
    ax2.vlines(3, matched_q1, matched_q3, color="k", linestyle="-", lw=5)
    ax2.vlines(2, midmatched_q1, midmatched_q3, color="k", linestyle="-", lw=5)
    ax2.vlines(1, unmatched_q1, unmatched_q3, color="k", linestyle="-", lw=5)
    ax2.vlines(3, matched_min, matched_max, color="k", linestyle="-", lw=1)
    ax2.vlines(2, midmatched_min, midmatched_max, color="k", linestyle="-", lw=1)
    ax2.vlines(1, unmatched_min, unmatched_max, color="k", linestyle="-", lw=1)

    [ks_stat, ks_pval] = scipy.stats.ks_2samp(
        stringmatched_values, stringunmatched_values
    )
    plt.xticks(
        (1, 2, 3),
        ("StringDB Edge < 0.05", "0.05 < StringDB Edge < 0.95", "StringDB Edge > 0.95"),
    )
    plt.subplots_adjust(bottom=0.15, wspace=0.05)
    plt.xlabel("Two-Sided, Two-Sample K-S Test: " + f"{ks_pval:.6}")
    plt.ylabel("Morphology Screen Edge Weight (Similarity Score)")

    plt.show()
    if path_to_save is not None:
        fig.savefig(path_to_save, dpi=300, bbox_inches="tight")

    print("Statistics of high-weight edges vs. unmatched edges:")
    print(scipy.stats.ks_2samp(stringmatched_values, stringunmatched_values))
    print("Statistics of mid-weight edges vs. unmatched edges:")
    print(scipy.stats.ks_2samp(stringmid_values, stringunmatched_values))
    print("Statistics of high-weight edges vs. mid-weight edges:")
    print(scipy.stats.ks_2samp(stringmatched_values, stringmid_values))

    print(
        "Number of unmatched edges, based on stringDB thresholding: "
        + str(len(stringunmatched_values))
    )
    print(
        "Number of mid-weight edges, based on stringDB thresholding: "
        + str(len(stringmid_values))
    )
    print(
        "Number of high-weight edges, based on stringDB thresholding: "
        + str(len(stringmatched_values))
    )

    return (
        scipy.stats.ks_2samp(stringmatched_values, stringunmatched_values),
        scipy.stats.ks_2samp(stringmid_values, stringunmatched_values),
        scipy.stats.ks_2samp(stringmatched_values, stringmid_values),
    )


def plot_volcano(
    plot_zscores: pd.array,
    plot_fdrs: pd.array,
    volcano_label_threshold: int,
    plot_title: str,
    xaxis: str,
    feature_labels: list,
) -> None:
    """
    Given a set of Z-scores and FDRs, this script creates a volcano plot of features that are differentially 'expressed' compared to control.

    Parameters
    ----------
    plot_zscores : pd.array
        Z-scores of features
    plot_fdrs : pd.array
        FDRs of features
    volcano_label_threshold : int
        threshold at which to include labels of genes (as -log10 of FDR-corrected pval)
    plot_title : str
        title of the plot
    xaxis : str
        x-axis label
    feature_labels : list
        feature labels
    """
    plt.figure(figsize=(18, 16))
    plt.scatter(
        plot_zscores, -np.log10(plot_fdrs.astype(float)), s=20, c="k", linewidth=1
    )  # edgecolors=(1,1,1,0))
    texts = []
    labelcutoff = np.sort(-np.log10(plot_fdrs.astype(float)))[
        -(volcano_label_threshold + 1)
    ]
    for x, y, s in zip(
        plot_zscores, -np.log10(plot_fdrs.astype(float)), feature_labels
    ):
        if y > labelcutoff:
            texts.append(plt.text(x, y, s, size=9))
    if xaxis == "z":
        plt.xlabel("Median Z-score", fontsize=22)
    elif xaxis == "kstat":
        plt.xlabel("K-S Statistic", fontsize=22)
    plt.ylabel("Benjamini-Hochberg of Kolmogorov-Smirnov, -log10(FDR)", fontsize=22)
    plt.title(plot_title)
    plt.axvline(x=0, color="k", linestyle="dashed")


def plot_feature_volcano(
    feature_pd: pd.DataFrame,
    plot_gene: str,
    control_gene: str = "intergenic",
    volcano_label_threshold: int = 50,
    xaxis: str = "z",
    path_to_save: Optional[str] = None,
):
    """
    Given a feature matrix parquet input and a gene to plot, this script creates a volcano plot of features
    that are differentially 'expressed' compared to control.
    Statistical test is KS 2-sample 2-tail test with Benjamini-Hochberg correction, and plotted as -log10 FDR.
    Fold change is plotted as median-centered Z-score of each feature.

    Parameters
    ----------
    feature_pq: Parquet
        The multiIndexed feature matrix.
    plot_gene: string
        The gene to plot.  Example: 'ACTB'
    control_gene: string
        The control gene for statistical comparison.  Example: 'intergenic'
    volcano_label_threshold: numeric
        The threshold at which to include labels of genes (as -log10 of FDR-corrected pval).  Note that high numbers will slow down processing.
    xaxis: string
        Whether to plot the Z-score or the KS Statistic along the X-axis.  String should be 'z' or 'kstat'
    save_name: string
        Name of the PDF or SVG which will be automatically saved upon completion. Recommended to end with '.pdf'
    """

    dg_data_norm_clean = feature_pd.copy(deep=True)

    gene_data = dg_data_norm_clean[
        dg_data_norm_clean.index.get_level_values("gene_id").isin([plot_gene])
    ]
    control_data = dg_data_norm_clean[
        dg_data_norm_clean.index.get_level_values("gene_id").isin([control_gene])
    ]

    agg_features_stats = pd.DataFrame(
        index=gene_data.columns, columns=["median", "pval", "fdr", "kstat", "mean"]
    )

    print(
        "Running 2-sample KS tests on all features between target and control gene..."
    )
    for feature in agg_features_stats.index:
        [ks_stat, ks_pval] = scipy.stats.ks_2samp(
            control_data[feature].values, gene_data[feature].values
        )
        agg_features_stats.at[feature, "pval"] = ks_pval
        agg_features_stats.at[feature, "kstat"] = ks_stat
    agg_features_stats["fdr"] = multi.multipletests(
        agg_features_stats["pval"].values, method="fdr_bh"
    )[1]

    if xaxis == "z":
        for feature in agg_features_stats.index:
            agg_features_stats.at[feature, "median"] = np.median(
                gene_data[feature], axis=0
            )
        plot_xcoords = agg_features_stats["median"]
    elif xaxis == "kstat":
        for feature in agg_features_stats.index:
            agg_features_stats.loc[feature, "mean"] = np.mean(
                gene_data[feature], axis=0
            ) - np.mean(control_data[feature], axis=0)
        xdirection = np.array([agg_features_stats["mean"] > 0]).astype(int)[0] * 2 - 1

        plot_xcoords = agg_features_stats["kstat"] * xdirection

    plot_ycoords = agg_features_stats["fdr"]

    plot_volcano(
        plot_zscores=plot_xcoords,
        plot_fdrs=plot_ycoords,
        volcano_label_threshold=volcano_label_threshold,
        plot_title=plot_gene,
        xaxis=xaxis,
        feature_labels=dg_data_norm_clean.columns.values,
    )

    if path_to_save is not None:
        plt.savefig(path_to_save, dpi=300, bbox_inches="tight")


def plot_volcano_gene(
    plot_xcoords: pd.array,
    plot_fdrs: np.ndarray,
    volcano_label_threshold: float,
    plot_title: str,
    xaxis: str = "z",
    control_dot: Optional[str] = None,
):
    """
    Given a set of Z-scores and FDRs, this script creates a volcano plot of genes that are differentially 'expressed' compared to control.

    Parameters
    ----------
    plot_xcoords : pd.array
        _description_
    plot_fdrs : np.ndarray
        _description_
    volcano_label_threshold : float
        _description_
    plot_title : str
        _description_
    xaxis : str, optional
        _description_, by default "z"
    control_dot : Optional[str], optional
        _description_, by default None
    """
    plt.figure(figsize=(18, 16))
    plt.scatter(
        plot_xcoords, -np.log10(plot_fdrs.astype(float)), s=27, c="k", linewidth=0.5
    )

    if control_dot is not None:
        plt.scatter(
            plot_xcoords[control_dot],
            -np.log10(plot_fdrs[control_dot]),
            s=20,
            c="orange",
            edgecolor="k",
            linewidth=0.5,
        )

    texts = []
    for x, y, s in zip(
        plot_xcoords, -np.log10(plot_fdrs.astype(float)), plot_xcoords.index
    ):
        if y > volcano_label_threshold:
            texts.append(plt.text(x, y, s, size=9))

    plt.xlabel(xaxis)
    plt.ylabel("k-s (bh fdr)")
    plt.axhline(1.3)
    plt.axvline(0)
    plt.title(plot_title)


def plot_gene_volcano(
    feature_pd: pd.DataFrame,
    plot_feature: str = "nucleus_mask_area_pixel_sq",
    control_gene: str = "nontargeting",
    volcano_label_threshold: float = 2.5,
    xaxis: str = "z",
    control_dot: Optional[str] = "intergenic",
    path_to_save: Optional[str] = None,
):
    """
    Given a feature matrix parquet input and a feature to plot, this script creates a volcano plot of genes that differentially 'express' the given feature.
    Statistical test is KS 2-sample 2-tail test with Benjamini-Hochberg correction, and plotted as -log10 FDR.
    Fold change is plotted as median-centered Z-score of each gene.

    Parameters
    ----------
    feature_pq: Parquet
        The multiIndexed feature matrix.
    plot_feature: string
        The feature to plot.  Example: 'nucleus_mask_area_pixel_sq'
    control_gene: string
        The gene against which all other genes will be tested (K-S).  Often 'nontargeting' or 'intergenic'
    volcano_label_threshold: numeric
        The threshold at which to include labels of genes (as -log10 of FDR-corrected pval).  Note that low numbers will slow down processing.
    xaxis: string
        A string of 'z' or 'kstat' that determines what is plotted at the x-axis
    control_dot: string
        The gene (or control) to be labeled yellow; typically 'intergenic' or 'nontargeting'
    save_name: string
        Name of the PDF of SVG which will be automatically saved upon completion
    """

    feat_data = feature_pd.copy(deep=True)[[plot_feature]]

    agg_gene_features = pd.DataFrame(
        index=feat_data.index.get_level_values("gene_id").unique(),
        columns=["median", "pval", "fdr", "kstat", "mean"],
    )

    control_data = feat_data[
        feat_data.index.get_level_values("gene_id") == control_gene
    ]

    for gene in agg_gene_features.index:
        test_data = feat_data[feat_data.index.get_level_values("gene_id") == gene]
        [ks_stat, ks_pval] = scipy.stats.ks_2samp(
            control_data[plot_feature].values, test_data[plot_feature].values
        )
        agg_gene_features.loc[gene, "pval"] = ks_pval
        agg_gene_features.loc[gene, "kstat"] = ks_stat

    agg_gene_features["fdr"] = multi.multipletests(
        agg_gene_features["pval"], method="fdr_bh"
    )[1]

    if xaxis == "z":
        for gene in agg_gene_features.index:
            agg_gene_features["median"].loc[gene] = np.median(
                feat_data[feat_data.index.get_level_values("gene_id") == gene][
                    plot_feature
                ],
                axis=0,
            )
        plot_xcoords = agg_gene_features["median"]
    elif xaxis == "kstat":
        agg_gene_features["mean"] = feat_data.groupby("gene_id")[
            plot_feature
        ].mean() - np.mean(
            feat_data[feat_data.index.get_level_values("gene_id") == control_gene][
                plot_feature
            ],
            axis=0,
        )
        xdirection = np.array([agg_gene_features["mean"] > 0]).astype(int)[0] * 2 - 1
        plot_xcoords = agg_gene_features["kstat"] * xdirection

    plot_fdrs = agg_gene_features["fdr"]

    plot_volcano_gene(
        plot_xcoords=plot_xcoords,
        plot_fdrs=plot_fdrs,
        volcano_label_threshold=volcano_label_threshold,
        plot_title=plot_feature,
        xaxis=xaxis,
        control_dot=control_dot,
    )

    if path_to_save is not None:
        plt.savefig(path_to_save, dpi=300, bbox_inches="tight")


def plot_feature_histogram(
    feature_pd: pd.DataFrame,
    feature: str,
    test_gene: str,
    control_gene: str = "intergenic",
    bins: int = 80,
    path_to_save: Optional[str] = None,
):
    """
    Plots a feature histogram of two genes (a test and control gene), given an input .pq and the selected feature.

    Parameters
    ----------
    feature_pd: path
        Path to the input PQ file containing normalized, annotated, single cell data.
    feature: string
        The feature to be plotted
    test_gene: string
        The gene of interest to be plotted
    control_gene: string
        Control gene to be compared to in the histogram.  Typically 'intergenic' or 'nontargeting'
    bins: int
        Number of bins for the histogram
    path_to_save: string
        Name of the PDF or SVG which will be automatically saved upon completion. Recommended to end with '.pdf' or '.svg'
    """
    dg_data_norm_clean = feature_pd.copy(deep=True)

    plt.figure(figsize=(8, 4))
    plt.hist(
        dg_data_norm_clean[feature][
            dg_data_norm_clean.index.get_level_values("gene_id") == control_gene
        ],
        bins,
        density=True,
        alpha=0.3,
    )
    plt.hist(
        dg_data_norm_clean[feature][
            dg_data_norm_clean.index.get_level_values("gene_id") == test_gene
        ],
        bins,
        density=True,
        alpha=0.3,
    )
    plt.legend([control_gene, test_gene])

    # noise = np.random.normal(0, 1, (1000, ))
    density = stats.gaussian_kde(
        dg_data_norm_clean[feature][
            dg_data_norm_clean.index.get_level_values("gene_id") == control_gene
        ]
    )
    n, x, _ = plt.hist(
        dg_data_norm_clean[feature][
            dg_data_norm_clean.index.get_level_values("gene_id") == control_gene
        ],
        bins,
        histtype="step",
        density=True,
        color="blue",
    )
    plt.plot(x, density(x))

    density = stats.gaussian_kde(
        dg_data_norm_clean[feature][
            dg_data_norm_clean.index.get_level_values("gene_id") == test_gene
        ]
    )
    n, x, _ = plt.hist(
        dg_data_norm_clean[feature][
            dg_data_norm_clean.index.get_level_values("gene_id") == test_gene
        ],
        bins,
        histtype="step",
        density=True,
        color="orange",
    )
    plt.plot(x, density(x))
    plt.title(feature)
    if path_to_save is not None:
        plt.savefig(path_to_save, dpi=300, bbox_inches="tight")
    plt.show()


def plot_roc_curve(
    results: Dict[str, Any],
    stringdb_threshold: float,
    path_to_save: Optional[str] = None,
):
    """
    plot ROC curve

    Parameters
    ----------
    results : Dict[str, Any]
        results dictionary
    stringdb_threshold : float
        stringdb threshold
    path_to_save : Optional[str], optional
        path to save plot to, by default None
    """
    plt.figure(figsize=(10, 10))
    plt.plot(results["roc_fprs"], results["roc_tprs"])
    plt.plot([0, 1], [0, 1], "k:")
    plt.title(
        "Stringdb Threshold = "
        + str(stringdb_threshold)
        + ",  AUC = "
        + str(results["roc_auc"])
    )

    # Add annotation of TPR at target FPR
    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.05])

    plt.plot(
        [0.05, 0.05],
        [plt.ylim()[0], results["roc_tpr_at_5pc_fpr"]],
        color="r",
        linestyle="dotted",
    )
    plt.plot(
        [plt.xlim()[0], 0.05],
        [results["roc_tpr_at_5pc_fpr"], results["roc_tpr_at_5pc_fpr"]],
        color="r",
        linestyle="dotted",
    )
    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.05])

    plt.annotate(
        str(results["roc_tpr_at_5pc_fpr"]), (0.05, results["roc_tpr_at_5pc_fpr"])
    )
    if path_to_save is not None:
        plt.savefig(path_to_save, dpi=300, bbox_inches="tight")
    plt.show()
