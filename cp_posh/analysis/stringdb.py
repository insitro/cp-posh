import os
from typing import List
import pandas as pd


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
    curr_dir = "./"
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
