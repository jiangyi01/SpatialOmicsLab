import enum
import json

import requests


def process_bio_retrieval_ducoment(documents_df):
    ir_corpus = {}
    corpus2tool = {}
    for row in documents_df.itertuples():
        doc = row.document_content
        ir_corpus[row.docid] = (
            (doc.get("name", "") or "")
            + ", "
            + (doc.get("description", "") or "")
            + ", "
            + (doc.get("url", "") or "")
            + ", "
            + ", required_params: "
            + json.dumps(doc.get("required_parameters", ""))
            + ", optional_params: "
            + json.dumps(doc.get("optional_parameters", ""))
        )

        corpus2tool[
            (doc.get("name", "") or "")
            + ", "
            + (doc.get("description", "") or "")
            + ", "
            + (doc.get("url", "") or "")
            + ", "
            + ", required_params: "
            + json.dumps(doc.get("required_parameters", ""))
            + ", optional_params: "
            + json.dumps(doc.get("optional_parameters", ""))
        ] = doc.get("name", "") or ""
    return ir_corpus, corpus2tool


def parse_hpo_obo(file_path):
    """Parse the HPO OBO file and create a dictionary mapping HP IDs to phenotype descriptions.

    Args:
        file_path (str): Path to the HPO OBO file.

    Returns:
        dict: A dictionary where keys are HP IDs and values are phenotype descriptions.

    """
    hp_dict = {}
    current_id = None
    current_name = None

    with open(file_path) as file:
        for line in file:
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                # Any stanza header ([Term], [Typedef], ...) closes the previous term. Only
                # [Term] stanzas carry an "id: HP:" line, so without resetting here a trailing
                # [Typedef]'s "name:" would overwrite the last term's name while its HP id is
                # still current -- silently corrupting that phenotype's label at EOF.
                if current_id and current_name:
                    hp_dict[current_id] = current_name
                current_id = None
                current_name = None
            elif line.startswith("id: HP:"):
                current_id = line.split(": ")[1]
            elif line.startswith("name:"):
                current_name = line.split(": ", 1)[1]

        # Add the last term to the dictionary
        if current_id and current_name:
            hp_dict[current_id] = current_name

    return hp_dict


class ID(enum.Enum):
    ENTREZ = "Entrez"
    ENSEMBL = "Ensembl without version"  # e.g. ENSG00000123374
    ENSEMBL_W_VERSION = "Ensembl with version"  # e.g. ENSG00000123374.10 (needed for GTEx)


def get_gene_id(gene_symbol: str, id_type: ID):
    """Get the ID for a gene symbol. If no match found, returns None."""
    if id_type == ID.ENTREZ:
        return _get_gene_id_entrez(gene_symbol)
    elif id_type == ID.ENSEMBL:
        return _get_gene_id_ensembl(gene_symbol)
    elif id_type == ID.ENSEMBL_W_VERSION:
        return _get_gene_id_ensembl_with_version(gene_symbol)
    else:
        raise ValueError(f"ID type {id_type} not supported")


def _get_gene_id_entrez(gene_symbol: str):
    """Get the Entrez ID for a gene symbol. If no match found, returns None
    e.g. 1017 (CDK2).
    """
    api_call = f"https://mygene.info/v3/query?species=human&q=symbol:{gene_symbol}"
    response = requests.get(api_call)
    response_json = response.json()

    if len(response_json["hits"]) == 0:
        return None
    else:
        return response_json["hits"][0]["entrezgene"]


def _get_gene_id_ensembl(gene_symbol):
    """Get the Ensembl ID for a gene symbol. If no match found, returns None
    e.g. ENSG00000123374.
    """
    api_call = f"https://mygene.info/v3/query?species=human&fields=ensembl&q=symbol:{gene_symbol}"
    response = requests.get(api_call)
    response_json = response.json()

    if len(response_json["hits"]) == 0:
        return None
    else:
        ensembl = response_json["hits"][0]["ensembl"]
        if isinstance(ensembl, list):
            return ensembl[0][
                "gene"
            ]  # Sometimes returns a list, for example RNH1 (first elem is on chr11, second is on scaffold_hschr11)
        else:
            return ensembl["gene"]


def _get_gene_id_ensembl_with_version(gene_symbol):
    """Get the Ensembl ID for a gene symbol. If no match found, returns None
    e.g. ENSG00000123374.10.
    """
    api_base = "https://gtexportal.org/api/v2/reference/gene"
    params = {"geneId": gene_symbol}
    response_json = requests.get(api_base, params=params).json()

    if len(response_json["data"]) == 0:
        return None
    else:
        return response_json["data"][0]["gencodeId"]
