"""Gene, variant and chemical identifier services: MyGene/MyVariant/MyChem, TogoID, BridgeDb, HGNC.

Every identifier in a spatial-omics analysis arrives in somebody else's vocabulary. A Visium
``var_names`` index is Ensembl gene IDs, a marker panel from a paper is HGNC symbols, a drug target
list is DrugBank accessions, and a GWAS hit is an rsID. The tools here are the translation layer
between those vocabularies, plus the annotation lookups that hang off them.

Four services, chosen because each answers a question the others cannot:

* **BioThings** (``mygene.info``/``myvariant.info``/``mychem.info``) -- free-text search and full
  annotation records for genes, variants and chemicals.
* **TogoID** (``api.togoid.dbcls.jp``) -- bulk conversion between 100+ identifier namespaces.
* **BridgeDb** (``webservice.bridgedb.org``) -- cross-references for metabolites and lipids as well
  as genes, which is where TogoID's coverage thins out.
* **HGNC** (``rest.genenames.org``) -- the naming authority itself, and the only one that will tell
  you a symbol was *renamed* rather than simply not found.

Every outbound call goes through :mod:`spatialomicsgym.utils.http_client`, which enforces HTTPS, an
explicit host allowlist, one shared connection pool, a timeout and bounded retry. No function here
calls ``requests`` directly.

Return shape, uniform across the module: ``{"status": "success", "data": ..., "metadata": {...}}``
on success and ``{"status": "error", "error": "<what went wrong and what to do about it>"}`` on
failure. Nothing raises for an expected failure -- a network problem or a bad argument comes back as
a value, so one failed lookup inside a longer script does not abort the rest of it. The agent loop
recognises ``"status": "error"`` as a failed action, so a failure is never silently read as data.

These functions return a dict; ``print()`` the result (or the part of it you need) or it will not
appear in the observation.

------------------------------------------------------------------------------------------------
Adapted from ToolUniverse -- https://github.com/mims-harvard/ToolUniverse -- at commit
``f075c2a75e8b35ae5dbb220d48d4e87e980388b1``. Copyright [2025] [ToolUniverse team], licensed under
the Apache License, Version 2.0.

CHANGED BY SPATIALOMICSGYM, as Apache-2.0 section 4(b) requires this file to state. The endpoint
knowledge, the argument semantics and the response-repair behaviour (HGNC retired-symbol
resolution, MyVariant assembly labelling and sibling-record disclosure) are upstream's; the code is
not. Specifically: the ``BaseTool``/``register_tool``/config-driven ``operation`` dispatch machinery
was not vendored and each upstream operation is re-expressed here as a plain function; raw
``requests`` calls were replaced by our HTTP layer; user-supplied values interpolated into URL
*paths* are percent-encoded, which upstream does not do; and the return payloads carry our
``metadata`` conventions. See ``VENDORING.md`` for the full record.
"""

import inspect
import json
import logging
from urllib.parse import quote

from spatialomicsgym.utils.http_client import HttpError, request_json, request_text

logger = logging.getLogger(__name__)

# Every host this module is permitted to reach. Passed to the HTTP layer on every call, which
# refuses anything else -- so no argument, however malformed or adversarial, can redirect a request
# at a host that is not on this list. All six cleared under the RL-1 origin review recorded in
# CHINA_EXCLUSION.md (Scripps Research US, DBCLS Japan, BridgeDb/Maastricht NL, EMBL-EBI UK).
_ALLOWED_HOSTS = (
    "mygene.info",
    "myvariant.info",
    "mychem.info",
    "api.togoid.dbcls.jp",
    "webservice.bridgedb.org",
    "rest.genenames.org",
)

_MYGENE_BASE = "https://mygene.info/v3"
_MYVARIANT_BASE = "https://myvariant.info/v1"
_MYCHEM_BASE = "https://mychem.info/v1"
_TOGOID_BASE = "https://api.togoid.dbcls.jp"
_BRIDGEDB_BASE = "https://webservice.bridgedb.org"
_HGNC_BASE = "https://rest.genenames.org"

#: BioThings caps a page at 1000 but starts truncating usefulness long before that; upstream caps
#: at 100 and we keep the same ceiling so a stray ``size=100000`` cannot paste a megabyte into the
#: observation.
_MAX_PAGE = 100

#: MyGene.info answers at most 1000 terms per batch POST, and a whole ``var_names`` index is tens of
#: thousands, so ``mygene_batch_query`` sends it in batches of this size and joins the answers in
#: input order (hunt 2026-09-30, uT3-atlases-4).
_MYGENE_BATCH = 1000

#: TogoID takes its ids in the GET query string, where a whole ``var_names`` index (~300 kB) is far
#: past what any server accepts in a URL. Ids are sent in groups whose comma-joined length stays
#: under this many characters (hunt 2026-09-30, uT3-atlases-4).
_TOGOID_QUERY_CHARS = 4000

#: MyVariant serves genomic coordinates in an assembly its payload never names, and the one it
#: defaults to is GRCh37/hg19: rs4244285 is chr10:g.96541616G>C there and chr10:g.94781859G>C on
#: GRCh38, ~1.76 Mb apart. Every response from the variant tools therefore carries an explicit
#: ``coordinate_assembly``, spelled both ways because clinicians write GRCh37 where pipelines write
#: hg19.
_ASSEMBLY_LABELS = {"hg19": "hg19 (GRCh37)", "hg38": "hg38 (GRCh38)"}

#: Top-level MyVariant fields that locate or identify a variant rather than annotate it. A record
#: carrying only these is a stub -- the state the sibling-record cross-check below looks for.
_IDENTITY_FIELD_ROOTS = frozenset(
    {"_id", "_score", "_version", "_license", "dbsnp", "chrom", "vcf", "hg19", "hg38", "observed"}
)

#: BridgeDb addresses a database by a one-to-three character system code. Callers say "Ensembl";
#: the service wants "En".
_BRIDGEDB_SYSTEM_CODES = {
    "HMDB": "Ch",
    "ChEBI": "Ce",
    "KEGG Compound": "Ck",
    "KEGG Drug": "Kd",
    "PubChem-compound": "Cpc",
    "Wikidata": "Wd",
    "CAS": "Ca",
    "Chemspider": "Cs",
    "Ensembl": "En",
    "HGNC": "H",
    "UniProt": "S",
    "NCBI Gene": "L",
    "RefSeq": "Q",
    "KEGG Genes": "Kg",
    "PDB": "Pd",
    "GeneOntology": "T",
    "InChIKey": "Ik",
    "SwissLipids": "Sl",
    "KNApSAcK": "Kn",
    "Rhea": "Rh",
    "MetaCyc": "Mc",
}
_BRIDGEDB_CODE_TO_NAME = {code: name for name, code in _BRIDGEDB_SYSTEM_CODES.items()}

#: HGNC fields that hold a *retired* name for a gene whose record now lives under a different
#: symbol. Ordered by strength of relation: an official rename outranks an informal alias.
_HGNC_SYMBOL_FALLBACKS = (("prev_symbol", "a", "previous symbol"), ("alias_symbol", "an", "alias symbol"))


# --------------------------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------------------------


def _error(message, **extra):
    """A failure the agent loop will recognise as a failed action rather than as data."""
    payload = {"status": "error", "error": message}
    payload.update(extra)
    return payload


def _ok(data, **extra):
    """A success payload in this module's uniform shape."""
    payload = {"status": "success", "data": data}
    payload.update(extra)
    return payload


def _missing(name, hint, *, plural=False):
    return _error(f"{name} {'are' if plural else 'is'} required. {hint}")


def _seg(value):
    """Percent-encode a caller-supplied value that is going into a URL *path*.

    ``safe=""`` so a slash in an identifier stays part of the identifier instead of becoming a path
    separator. Upstream interpolates these values raw.
    """
    return quote(str(value), safe="")


def _fetch_json(url, **kwargs):
    """``(payload, None)`` on success, ``(None, error_payload)`` on any HTTP failure."""
    try:
        return request_json(url, allowed_hosts=_ALLOWED_HOSTS, **kwargs), None
    except HttpError as exc:
        return None, _error(exc.detail)


def _page_size(size):
    try:
        return max(1, min(int(size), _MAX_PAGE))
    except (TypeError, ValueError):
        return 10


def _id_list(ids):
    """Caller-supplied identifiers as a list of clean strings, whatever container they came in.

    These tools are called from agent code, and ``adata.var_names`` -- a pandas Index -- is the input
    the descriptions invite. ``if not ids`` raised "truth value of an Index is ambiguous" on it, and
    anything that was not a list, tuple or set was ``str()``-ed whole, so a one-element array went
    out as the query "['CDK2']" (hunt 2026-09-30, uT3-atlases-4). A string is split on commas, any
    other iterable is read element by element, and anything else is one id.
    """
    if ids is None:
        return []
    if isinstance(ids, str):
        items = ids.split(",")
    else:
        try:
            items = list(ids)
        except TypeError:
            items = [ids]
    return [text for text in (str(item).strip() for item in items if item is not None) if text]


# --------------------------------------------------------------------------------------------
# MyGene.info -- genes
# --------------------------------------------------------------------------------------------


def mygene_query_genes(
    query,
    species="human",
    fields="symbol,name,entrezgene,ensembl.gene,summary",
    size=10,
):
    """Search MyGene.info for genes by symbol, name, keyword or identifier.

    This is the "I have a string, which gene is it" tool. It accepts free text
    ("kinase involved in apoptosis"), a symbol ("CDK2"), an identifier ("ENSG00000123374") or a
    fielded Lucene query ("symbol:BRCA* AND taxid:9606").

    Parameters
    ----------
    query (str, required): Search string. Free text, a symbol, an identifier, or fielded Lucene
        syntax such as "symbol:CDK2" or "ensembl.gene:ENSG00000123374".
    species (str): Species name or taxonomy ID, e.g. "human", "mouse", or "9606". Default "human".
    fields (str): Comma-separated annotation fields to return. Dotted paths select subfields.
    size (int): Maximum hits to return, capped at 100.

    Returns
    -------
    dict: ``{"status": "success", "data": {"hits": [...], "total": N}}`` or a ``status: error``
        payload. ``data`` is MyGene's own response, unmodified.

    Examples
    --------
    - mygene_query_genes("CDK2")
    - mygene_query_genes("symbol:GFAP", species="mouse", fields="symbol,name,entrezgene")
    """
    if not query:
        return _missing("query", "Pass a gene symbol, an identifier, or free text to search for.")
    payload, failure = _fetch_json(
        f"{_MYGENE_BASE}/query",
        params={"q": query, "species": species, "fields": fields, "size": _page_size(size)},
    )
    return failure or _ok(payload, metadata={"source": "MyGene.info", "query": query, "species": species})


def mygene_get_gene_annotation(
    gene_id,
    fields="symbol,name,entrezgene,ensembl,summary,go,pathway,interpro",
):
    """Retrieve the full annotation record for one gene, by identifier.

    Use this once you already know which gene you mean. ``mygene_query_genes`` is the tool that
    turns a name into an identifier; this one turns an identifier into everything known about it,
    including GO terms, pathway memberships, InterPro domains and the RefSeq summary paragraph.

    Parameters
    ----------
    gene_id (str, required): Entrez Gene ID ("1017"), Ensembl gene ID ("ENSG00000123374"), or any
        other identifier MyGene indexes as a primary key.
    fields (str): Comma-separated annotation fields. Use "all" for the complete record, which is
        large.

    Returns
    -------
    dict: ``{"status": "success", "data": {...}}`` where ``data`` is the gene record, or a
        ``status: error`` payload.

    Examples
    --------
    - mygene_get_gene_annotation("1017")
    - mygene_get_gene_annotation("ENSG00000123374", fields="symbol,name,summary,go.BP")
    """
    if not gene_id:
        return _missing(
            "gene_id",
            "Pass an Entrez ID such as '1017' or an Ensembl ID such as 'ENSG00000123374'. "
            "Use mygene_query_genes to find one from a symbol or a description.",
        )
    payload, failure = _fetch_json(f"{_MYGENE_BASE}/gene/{_seg(gene_id)}", params={"fields": fields})
    return failure or _ok(payload, metadata={"source": "MyGene.info", "gene_id": str(gene_id)})


def mygene_batch_query(gene_ids, species="human", fields="symbol,name,entrezgene,ensembl.gene"):
    """Annotate many genes in one request.

    The tool to reach for when translating a whole ``var_names`` index or a marker-gene list. One
    POST handles the batch, so a 500-gene panel costs one round trip instead of 500. Identifiers
    are matched against Entrez ID, Ensembl gene ID and symbol simultaneously, so a mixed list works.

    Parameters
    ----------
    gene_ids (list, required): Gene identifiers -- symbols, Entrez IDs, Ensembl IDs, or a mix. A
        comma-separated string, a pandas Index (``adata.var_names``) or an array is also accepted;
        more than 1000 ids are sent in batches of 1000.
    species (str): Species name or taxonomy ID. Default "human".
    fields (str): Comma-separated annotation fields to return for each gene.

    Returns
    -------
    dict: ``{"status": "success", "data": {"results": [...]}}`` -- one entry per input, in input
        order, each carrying a ``query`` key echoing what was asked for. Unmatched inputs come back
        with ``"notfound": true`` rather than being dropped, so the list stays aligned.

    Examples
    --------
    - mygene_batch_query(["CDK2", "BRCA1", "GFAP"])
    - mygene_batch_query(["ENSG00000123374", "ENSG00000012048"], fields="symbol,name")
    """
    ids = _id_list(gene_ids)
    if not ids:
        return _missing("gene_ids", "Pass a list of gene symbols or identifiers, e.g. ['CDK2', 'BRCA1'].")
    results = []
    for start in range(0, len(ids), _MYGENE_BATCH):
        batch = ids[start : start + _MYGENE_BATCH]
        payload, failure = _fetch_json(
            f"{_MYGENE_BASE}/query",
            method="POST",
            form_data={
                "q": ",".join(batch),
                "scopes": "entrezgene,ensembl.gene,symbol",
                "species": species,
                "fields": fields,
            },
        )
        if failure:
            if start:
                failure["error"] += (
                    f" (failed on ids {start + 1}-{start + len(batch)} of {len(ids)}; nothing from the "
                    "earlier batches is returned, so retry the whole call)"
                )
            return failure
        if isinstance(payload, list):
            results.extend(payload)
        else:
            results.append(payload)
    metadata = {"source": "MyGene.info", "requested": len(ids), "species": species}
    if len(ids) > _MYGENE_BATCH:
        metadata["batches"] = -(-len(ids) // _MYGENE_BATCH)
    return _ok({"results": results}, metadata=metadata)


# --------------------------------------------------------------------------------------------
# MyVariant.info -- variants
# --------------------------------------------------------------------------------------------


def _resolve_assembly(assembly):
    """``(assembly, label, None)`` or ``(None, None, error_payload)``."""
    name = str(assembly or "hg19").strip().lower()
    if name not in _ASSEMBLY_LABELS:
        return (
            None,
            None,
            _error(f"Unknown assembly '{name}'. Supported assemblies: {', '.join(sorted(_ASSEMBLY_LABELS))}."),
        )
    return name, _ASSEMBLY_LABELS[name], None


def _assembly_params(assembly):
    """MyVariant already treats a missing ``assembly`` as hg19, so omitting it keeps the default
    request -- and therefore the default response -- identical to one made without the parameter."""
    return {} if assembly == "hg19" else {"assembly": assembly}


def _is_rsid(variant_id):
    text = str(variant_id).strip().lower()
    return text.startswith("rs") and text[2:].isdigit()


def _annotation_roots(fields, fallback):
    """Which of the requested field roots would carry an annotation rather than an identifier.

    "all"/"*" name no field in particular, so they fall back to the tool's declared list: asking
    MyVariant for everything is still asking it for the scores.
    """
    text = str(fields or "").strip().lower()
    if text in ("", "all", "*"):
        text = fallback.lower()
    roots = {part.strip().split(".")[0] for part in text.split(",")}
    return {root for root in roots if root} - _IDENTITY_FIELD_ROOTS


def _sibling_records(payload, variant_id, fields, assembly, fallback_fields):
    """Surface records filed under the same rsID that do carry the requested annotation.

    MyVariant can hold several records for one rsID and ``/variant/<rsid>`` answers with the
    highest-scoring one. For rs267606617 that is chrMT:m.1555A>G, a dbSNP-only stub, while the CADD
    score lives on the sibling chrMT:g.1555A>G -- so a pathogenicity lookup reports "no scores" for
    a variant that has one. The siblings are reported *alongside* the primary record, never in place
    of it: ``data`` stays exactly what MyVariant resolved.

    The extra request is confined to the case that motivates it, so the common path costs nothing.
    A failure here returns nothing at all -- a cross-check is a bonus and must not fail the answer.
    """
    if not _is_rsid(variant_id):
        return {}
    wanted = _annotation_roots(fields, fallback_fields)
    if not wanted:
        return {}

    primary = payload if isinstance(payload, list) else [payload]
    present = [set(record) for record in primary if isinstance(record, dict)]
    if any(roots & wanted for roots in present):
        return {}

    found = myvariant_query_variants(query=variant_id, fields=fields, size=10, assembly=assembly)
    try:
        hits = found["data"]["hits"]
    except (KeyError, TypeError):
        return {}

    primary_ids = {record.get("_id") for record in primary if isinstance(record, dict)}
    siblings = [
        hit for hit in hits if isinstance(hit, dict) and hit.get("_id") not in primary_ids and set(hit) & wanted
    ]
    if not siblings:
        return {}

    sibling_ids = [hit["_id"] for hit in siblings]
    resolved = ", ".join(sorted(str(i) for i in primary_ids if i)) or "a record"
    logger.info(
        "MyVariant %s resolved to a record without %s; disclosing %d sibling record(s)",
        variant_id,
        ", ".join(sorted(wanted)),
        len(siblings),
    )
    return {
        "sibling_variant_ids_with_requested_fields": sibling_ids,
        "sibling_records_with_requested_fields": siblings,
        "sibling_record_note": (
            f"MyVariant.info files more than one record under {variant_id}. /variant/{variant_id} "
            f"resolves to the highest-scoring one ({resolved}), which carries none of the requested "
            f"{', '.join(sorted(wanted))} fields, while {', '.join(str(i) for i in sibling_ids)} does. "
            "'data' is left exactly as MyVariant returned it; the record(s) carrying the requested "
            "fields are in 'sibling_records_with_requested_fields'."
        ),
    }


def _variant_404_hint(variant_id, label):
    """``/variant/<hgvs>`` is a verbatim key lookup, not a liftover, so an assembly/id mismatch is a
    404 rather than a translated answer. A bare "request failed" sends the caller hunting for an
    outage instead of for the id they actually needed."""
    if _is_rsid(variant_id):
        return ""
    return (
        f" No record with id '{variant_id}' exists in {label}. MyVariant looks HGVS ids up verbatim "
        "and never lifts coordinates over, so the id must already be written in the assembly you "
        "asked for. Supply the id for that assembly, switch `assembly`, or pass the rsID, which "
        "resolves in either."
    )


def myvariant_query_variants(
    query,
    fields="dbsnp.rsid,clinvar.rcv.clinical_significance,cadd.phred,gnomad_genome.af.af",
    size=10,
    assembly="hg19",
):
    """Search MyVariant.info for variants by rsID, gene, consequence or clinical significance.

    Accepts free text or fielded Lucene syntax, so it answers both "what is rs334" and "every
    pathogenic ClinVar variant in BRCA1".

    Parameters
    ----------
    query (str, required): Search string. An rsID ("rs334"), an HGVS id, or a fielded query such as
        "clinvar.gene.symbol:BRCA1 AND clinvar.rcv.clinical_significance:pathogenic".
    fields (str): Comma-separated annotation fields to return.
    size (int): Maximum hits to return, capped at 100.
    assembly (str): Reference assembly for the returned coordinates -- "hg19" (GRCh37, the default)
        or "hg38" (GRCh38).

    Returns
    -------
    dict: ``{"status": "success", "coordinate_assembly": "...", "data": {"hits": [...]}}`` or a
        ``status: error`` payload. ``coordinate_assembly`` names the frame the coordinates in
        ``data`` are expressed in, which MyVariant's own payload does not.

    Examples
    --------
    - myvariant_query_variants("rs334")
    - myvariant_query_variants("clinvar.gene.symbol:BRCA1", size=25, assembly="hg38")
    """
    if not query:
        return _missing("query", "Pass an rsID such as 'rs334', an HGVS id, or a fielded query.")
    assembly, label, failure = _resolve_assembly(assembly)
    if failure:
        return failure
    params = {"q": query, "fields": fields, "size": _page_size(size)}
    params.update(_assembly_params(assembly))
    payload, failure = _fetch_json(f"{_MYVARIANT_BASE}/query", params=params)
    if failure:
        failure["coordinate_assembly"] = label
        return failure
    return _ok(payload, coordinate_assembly=label, metadata={"source": "MyVariant.info", "query": query})


def _get_variant(variant_id, fields, assembly, hint_required, default_fields):
    """Shared body of the two ``/variant/<id>`` tools, which differ only in their default fields.

    ``default_fields`` is the calling tool's own declared ``fields`` default. The sibling cross-check
    needs it when the caller asked for "all": that names no field in particular, and passing the
    caller's "all" as the fallback made the check look for a field called ``all``, spend a second
    request and never disclose a sibling (hunt 2026-09-30, uT3-atlases-14).
    """
    if not variant_id:
        return _missing("variant_id", hint_required)
    assembly, label, failure = _resolve_assembly(assembly)
    if failure:
        return failure

    params = {"fields": fields}
    params.update(_assembly_params(assembly))
    url = f"{_MYVARIANT_BASE}/variant/{_seg(variant_id)}"
    try:
        payload = request_json(url, allowed_hosts=_ALLOWED_HOSTS, params=params)
    except HttpError as exc:
        detail = exc.detail
        if exc.status == 404:
            detail += _variant_404_hint(variant_id, label)
        return _error(detail, coordinate_assembly=label)

    result = _ok(payload, coordinate_assembly=label, metadata={"source": "MyVariant.info", "variant_id": variant_id})
    result.update(_sibling_records(payload, variant_id, fields, assembly, default_fields))
    return result


def _declared_fields(tool):
    """The ``fields`` default a public variant tool declares in its own signature."""
    return inspect.signature(tool).parameters["fields"].default


def myvariant_get_variant_annotation(
    variant_id,
    fields="dbsnp,clinvar,cadd,gnomad_genome,dbnsfp",
    assembly="hg19",
):
    """Retrieve the full annotation record for one variant, by rsID or HGVS identifier.

    Parameters
    ----------
    variant_id (str, required): rsID ("rs334") or HGVS genomic id ("chr11:g.5248232C>A"). An HGVS
        id must be written in the assembly named by ``assembly``; an rsID resolves in either.
    fields (str): Comma-separated annotation fields. "all" returns the complete record.
    assembly (str): "hg19" (GRCh37, the default) or "hg38" (GRCh38).

    Returns
    -------
    dict: ``{"status": "success", "coordinate_assembly": "...", "data": {...}}`` or a
        ``status: error`` payload. When MyVariant files several records under one rsID and the one
        it resolves to is an identifier-only stub, the records that do carry the requested fields
        appear in ``sibling_records_with_requested_fields`` beside ``data``, never instead of it.

    Examples
    --------
    - myvariant_get_variant_annotation("rs334")
    - myvariant_get_variant_annotation("chr7:g.140453136A>T", assembly="hg19")
    """
    return _get_variant(
        variant_id,
        fields,
        assembly,
        "Pass an rsID such as 'rs334' or an HGVS genomic id such as 'chr11:g.5248232C>A'.",
        _declared_fields(myvariant_get_variant_annotation),
    )


def myvariant_get_pathogenicity_scores(
    variant_id,
    fields=(
        "dbnsfp.revel.score,cadd.phred,dbnsfp.alphamissense.score,dbnsfp.alphamissense.pred,"
        "dbnsfp.sift.score,dbnsfp.sift.pred,dbnsfp.polyphen2.hdiv.score,dbnsfp.polyphen2.hdiv.pred,"
        "dbnsfp.metarnn.score,dbnsfp.metarnn.pred,cadd.gerp.rs,"
        "dbnsfp.phylop.100way_vertebrate.rankscore,dbnsfp.phastcons.100way_vertebrate.rankscore,"
        "dbnsfp.vest4.score,dbnsfp.mutationtaster.pred,clinvar.rcv.clinical_significance,dbsnp.rsid"
    ),
    assembly="hg19",
):
    """Retrieve pathogenicity prediction scores for one variant, from dbNSFP in a single call.

    Returns REVEL, CADD, AlphaMissense, SIFT, PolyPhen-2, MetaRNN, GERP, PhyloP and PhastCons
    together. Reach for this when ``myvariant_query_variants`` comes back without dbnsfp data, or
    when you specifically need REVEL/AlphaMissense for an ACMG PP3/BP4 call.

    Parameters
    ----------
    variant_id (str, required): rsID ("rs45478192") or HGVS genomic id ("chr16:g.23635348A>C"). An
        HGVS id must be written in the assembly named by ``assembly``; an rsID resolves in either.
    fields (str): Comma-separated fields, pre-set to the pathogenicity-score list.
    assembly (str): "hg19" (GRCh37, the default) or "hg38" (GRCh38).

    Returns
    -------
    dict: ``{"status": "success", "coordinate_assembly": "...", "data": {...}}`` or a
        ``status: error`` payload, with the same sibling-record disclosure as
        ``myvariant_get_variant_annotation``.

    Examples
    --------
    - myvariant_get_pathogenicity_scores("rs45478192")
    - myvariant_get_pathogenicity_scores("chr7:g.140453136A>T")
    """
    return _get_variant(
        variant_id,
        fields,
        assembly,
        "Pass an rsID such as 'rs45478192' or an HGVS genomic id such as 'chr16:g.23635348A>C'.",
        _declared_fields(myvariant_get_pathogenicity_scores),
    )


# --------------------------------------------------------------------------------------------
# MyChem.info -- chemicals and drugs
# --------------------------------------------------------------------------------------------


def mychem_query_chemicals(
    query,
    fields="drugbank.name,drugbank.drug_interactions,chebi,pubchem.cid,chembl.molecule_chembl_id",
    size=10,
):
    """Search MyChem.info for drugs and chemicals by name, InChIKey or identifier.

    Parameters
    ----------
    query (str, required): Drug or chemical name ("imatinib"), an InChIKey, or a fielded query such
        as "drugbank.name:aspirin".
    fields (str): Comma-separated annotation fields to return.
    size (int): Maximum hits to return, capped at 100.

    Returns
    -------
    dict: ``{"status": "success", "data": {"hits": [...]}}`` or a ``status: error`` payload.

    Examples
    --------
    - mychem_query_chemicals("imatinib")
    - mychem_query_chemicals("drugbank.name:aspirin", fields="drugbank,chebi,pubchem")
    """
    if not query:
        return _missing("query", "Pass a drug name such as 'imatinib', an InChIKey, or a fielded query.")
    payload, failure = _fetch_json(
        f"{_MYCHEM_BASE}/query",
        params={"q": query, "fields": fields, "size": _page_size(size)},
    )
    return failure or _ok(payload, metadata={"source": "MyChem.info", "query": query})


def mychem_get_chemical_annotation(chem_id, fields="drugbank,chebi,pubchem,chembl,drugcentral"):
    """Retrieve the full annotation record for one chemical, by InChIKey or database identifier.

    Parameters
    ----------
    chem_id (str, required): InChIKey (recommended, e.g. "KTUFNOKKBVMGRW-UHFFFAOYSA-N"), DrugBank
        accession, ChEMBL ID or PubChem CID.
    fields (str): Comma-separated annotation fields. "all" returns the complete record.

    Returns
    -------
    dict: ``{"status": "success", "data": {...}}`` or a ``status: error`` payload.

    Examples
    --------
    - mychem_get_chemical_annotation("KTUFNOKKBVMGRW-UHFFFAOYSA-N")
    - mychem_get_chemical_annotation("CHEMBL941", fields="chembl,drugbank")
    """
    if not chem_id:
        return _missing(
            "chem_id",
            "Pass an InChIKey such as 'KTUFNOKKBVMGRW-UHFFFAOYSA-N', or use mychem_query_chemicals "
            "to find one from a drug name.",
        )
    payload, failure = _fetch_json(f"{_MYCHEM_BASE}/chem/{_seg(chem_id)}", params={"fields": fields})
    return failure or _ok(payload, metadata={"source": "MyChem.info", "chem_id": str(chem_id)})


# --------------------------------------------------------------------------------------------
# TogoID -- bulk identifier conversion
# --------------------------------------------------------------------------------------------


def togoid_convert(ids, source, target):
    """Convert identifiers between namespaces in bulk, using TogoID.

    The workhorse for "my matrix is Ensembl and my marker list is UniProt". TogoID covers 100+
    namespaces; ``togoid_list_datasets`` enumerates them with their exact dataset keys.

    Source and target must be **directly related** in TogoID's graph -- pick adjacent dataset types,
    or convert in two steps through an intermediate. TogoID itself supports multi-hop routes, but
    this tool issues a direct two-step route, so an unrelated pair comes back as an explicit "no
    route" failure rather than an empty result you might mistake for "no matches".

    Parameters
    ----------
    ids (str, required): Identifiers to convert. Comma-separated, or a list, pandas Index or array.
    source (str, required): Source dataset key, e.g. "ensembl_gene", "hgnc", "uniprot".
    target (str, required): Target dataset key, e.g. "ncbigene", "pdb", "chebi".

    Returns
    -------
    dict: ``{"status": "success", "data": {"input_ids": [...], "source": ..., "target": ...,
        "converted_ids": [...]}}`` or a ``status: error`` payload. A pair with no route is reported
        as an error naming both datasets.

    Examples
    --------
    - togoid_convert("ENSG00000012048", source="ensembl_gene", target="uniprot")
    - togoid_convert(["P38398", "P04637"], source="uniprot", target="pdb")
    """
    id_list = _id_list(ids)
    if not id_list:
        return _missing("ids", "Pass one or more identifiers, e.g. 'ENSG00000012048' or a list of them.")
    if not source or not target:
        return _missing(
            "source and target",
            "Both are dataset keys, e.g. source='ensembl_gene', target='uniprot'. "
            "Call togoid_list_datasets to see the valid keys.",
            plural=True,
        )

    groups, group, length = [], [], 0
    for identifier in id_list:
        if group and length + 1 + len(identifier) > _TOGOID_QUERY_CHARS:
            groups.append(group)
            group, length = [], 0
        length += len(identifier) + (1 if group else 0)
        group.append(identifier)
    groups.append(group)

    converted = []
    for group in groups:
        params = {"ids": ",".join(group), "route": f"{source},{target}", "format": "json"}

        # TogoID answers "no route between these datasets" with HTTP 400 and a JSON body that
        # explains it. raise_for_status() would collapse the only actionable part of the response
        # into a status code, so read the body first and only then decide it is a failure.
        try:
            body = request_text(f"{_TOGOID_BASE}/convert", allowed_hosts=_ALLOWED_HOSTS, params=params)
        except HttpError as exc:
            message = _togoid_message(exc.body)
            if message:
                return _error(
                    f"TogoID cannot convert {source} to {target} directly: {message} "
                    "Pick adjacent dataset types, or convert via an intermediate dataset in two calls."
                )
            return _error(exc.detail)

        try:
            payload = json.loads(body)
        except ValueError:
            return _error(f"TogoID returned a body that is not JSON: {body[:200]}")

        found = payload.get("results", []) if isinstance(payload, dict) else []
        converted.extend(found if isinstance(found, list) else [found])

    metadata = {"source": "TogoID", "route": f"{source} -> {target}", "num_converted": len(converted)}
    if len(groups) > 1:
        metadata["requests"] = len(groups)
    return _ok(
        {
            "input_ids": id_list,
            "source": source,
            "target": target,
            "converted_ids": converted,
        },
        metadata=metadata,
    )


def _togoid_message(body):
    """TogoID's failure bodies carry the human-readable reason under ``message``."""
    try:
        parsed = json.loads(body or "")
    except ValueError:
        return ""
    return parsed.get("message", "") if isinstance(parsed, dict) else ""


def togoid_list_datasets(category=None):
    """List the identifier namespaces TogoID can convert between.

    Call this before ``togoid_convert`` when you are not certain of a dataset key -- the keys are
    exact strings ("ensembl_gene", not "Ensembl") and a wrong one is a failed conversion.

    Parameters
    ----------
    category (str, optional): Restrict to one category, case-insensitive, e.g. "gene", "protein",
        "chemical", "structure". Omit to list every dataset.

    Returns
    -------
    dict: ``{"status": "success", "data": [{"dataset": ..., "label": ..., "category": ...}, ...]}``
        sorted by category then dataset, or a ``status: error`` payload.

    Examples
    --------
    - togoid_list_datasets()
    - togoid_list_datasets(category="protein")
    """
    payload, failure = _fetch_json(f"{_TOGOID_BASE}/config/dataset")
    if failure:
        return failure

    datasets = []
    for key, meta in (payload or {}).items():
        if not isinstance(meta, dict):
            continue
        entry = {"dataset": key, "label": meta.get("label", ""), "category": meta.get("category", "")}
        if category and entry["category"].lower() != str(category).lower():
            continue
        datasets.append(entry)
    datasets.sort(key=lambda d: (d["category"], d["dataset"]))

    metadata = {"source": "TogoID", "count": len(datasets)}
    if category and not datasets:
        metadata["no_results_note"] = (
            f"No TogoID dataset is in category '{category}'. Call togoid_list_datasets() with no "
            "category to see every dataset and the categories that exist."
        )
    return _ok(datasets, metadata=metadata)


# --------------------------------------------------------------------------------------------
# BridgeDb -- cross-references across 45+ databases
# --------------------------------------------------------------------------------------------


def _bridgedb_code(source):
    """Resolve a database name or system code to a BridgeDb system code, passing through anything
    unrecognised so the service can report the error itself."""
    if source in _BRIDGEDB_CODE_TO_NAME:
        return source
    lowered = str(source).lower()
    for name, code in _BRIDGEDB_SYSTEM_CODES.items():
        if name.lower() == lowered:
            return code
    return str(source)


def _bridgedb_rows(text):
    """Parse BridgeDb's tab-separated ``identifier<TAB>database`` response."""
    rows = []
    for line in (text or "").strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            rows.append(
                {
                    "identifier": parts[0],
                    "database": parts[1],
                    "system_code": _BRIDGEDB_SYSTEM_CODES.get(parts[1], ""),
                }
            )
    return rows


def _bridgedb_text(url, params=None):
    """``(text, None)`` or ``(None, error_payload)``.

    BridgeDb answers "nothing matched" with 204 and an empty body rather than with an error, and the
    HTTP layer passes a 204 through as an empty string, so a miss arrives here as ``""`` and parses
    to an empty list. That is the answer, not a failure.
    """
    try:
        return request_text(url, allowed_hosts=_ALLOWED_HOSTS, params=params), None
    except HttpError as exc:
        return None, _error(exc.detail)


def bridgedb_xrefs(identifier, source, organism="Human", target_source=None):
    """Get every cross-reference BridgeDb holds for one identifier.

    BridgeDb's coverage extends past genes to metabolites and lipids (HMDB, ChEBI, KEGG Compound,
    SwissLipids), which is where TogoID thins out -- so this is the tool for "what else is this
    metabolite called".

    Parameters
    ----------
    identifier (str, required): The identifier to look up, e.g. "ENSG00000012048" or "HMDB0000122".
    source (str, required): The database the identifier comes from. Either a name ("Ensembl",
        "HMDB", "UniProt", "HGNC", "ChEBI") or a BridgeDb system code ("En", "Ch", "S", "H", "Ce").
    organism (str): Species, e.g. "Human", "Mouse", "Rat". Default "Human".
    target_source (str, optional): Restrict the answer to one target database, by name or system
        code. Omit to get every cross-reference.

    Returns
    -------
    dict: ``{"status": "success", "data": {"cross_references": [...], "count": N, ...}}`` or a
        ``status: error`` payload. An identifier with no cross-references returns an empty list and
        ``count: 0``, which is an answer rather than a failure.

    Examples
    --------
    - bridgedb_xrefs("ENSG00000012048", source="Ensembl")
    - bridgedb_xrefs("HMDB0000122", source="HMDB", target_source="ChEBI")
    """
    if not identifier or not source:
        return _missing(
            "identifier and source",
            "Both are required: source names the database the identifier belongs to, "
            "e.g. bridgedb_xrefs('ENSG00000012048', source='Ensembl').",
            plural=True,
        )
    url = f"{_BRIDGEDB_BASE}/{_seg(organism)}/xrefs/{_seg(_bridgedb_code(source))}/{_seg(identifier)}"
    params = {"dataSource": _bridgedb_code(target_source)} if target_source else None
    text, failure = _bridgedb_text(url, params)
    if failure:
        return failure
    rows = _bridgedb_rows(text)
    return _ok(
        {
            "query_identifier": identifier,
            "query_source": source,
            "organism": organism,
            "cross_references": rows,
            "count": len(rows),
        },
        metadata={"source": "BridgeDb", "system_code": _bridgedb_code(source)},
    )


def bridgedb_search(query, organism="Human"):
    """Search BridgeDb for identifiers matching a name or symbol, across every database it indexes.

    Use this when you have a label and do not know which namespace it belongs to -- the results say
    which database each hit came from.

    Parameters
    ----------
    query (str, required): Free-text name or symbol, e.g. "BRCA1" or "glucose".
    organism (str): Species, e.g. "Human", "Mouse", "Rat". Default "Human".

    Returns
    -------
    dict: ``{"status": "success", "data": {"results": [...], "count": N, ...}}`` or a
        ``status: error`` payload.

    Examples
    --------
    - bridgedb_search("BRCA1")
    - bridgedb_search("glucose", organism="Human")
    """
    if not query:
        return _missing("query", "Pass a name or symbol to search for, e.g. 'BRCA1'.")
    text, failure = _bridgedb_text(f"{_BRIDGEDB_BASE}/{_seg(organism)}/search/{_seg(query)}")
    if failure:
        return failure
    rows = _bridgedb_rows(text)
    return _ok(
        {"query": query, "organism": organism, "results": rows, "count": len(rows)},
        metadata={"source": "BridgeDb"},
    )


def bridgedb_attributes(identifier, source, organism="Human"):
    """Get the stored properties of one identifier -- symbol, full name, synonyms, chromosome.

    Complements ``bridgedb_xrefs``: that answers "what else is this called in other databases",
    this answers "what does this database say about it".

    Parameters
    ----------
    identifier (str, required): The identifier to describe, e.g. "ENSG00000012048".
    source (str, required): The database the identifier comes from, by name ("Ensembl", "HGNC") or
        BridgeDb system code ("En", "H").
    organism (str): Species, e.g. "Human", "Mouse", "Rat". Default "Human".

    Returns
    -------
    dict: ``{"status": "success", "data": {"attributes": {...}, ...}}`` or a ``status: error``
        payload. Repeated ``Synonym`` rows are collected into an ``attributes["Synonyms"]`` list.

    Examples
    --------
    - bridgedb_attributes("ENSG00000012048", source="Ensembl")
    - bridgedb_attributes("HMDB0000122", source="HMDB")
    """
    if not identifier or not source:
        return _missing(
            "identifier and source",
            "Both are required, e.g. bridgedb_attributes('ENSG00000012048', source='Ensembl').",
            plural=True,
        )
    url = f"{_BRIDGEDB_BASE}/{_seg(organism)}/attributes/{_seg(_bridgedb_code(source))}/{_seg(identifier)}"
    text, failure = _bridgedb_text(url)
    if failure:
        return failure

    # BridgeDb's attributes endpoint answers `name<TAB>value` -- "Type\tprotein_coding",
    # "Synonym\tbeta-D-Glucose". That is the opposite column order from /xrefs and /search, which
    # answer `identifier<TAB>database`. Upstream reads all three the same way, so its attribute
    # dicts come out inverted ({"protein_coding": "Type"}) and its Synonym branch, which tests the
    # second column, never fires. Verified against the live service; see VENDORING.md.
    attributes = {}
    synonyms = []
    for line in (text or "").strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            key, value = parts[0], parts[1]
            if key == "Synonym":
                synonyms.append(value)
            else:
                attributes[key] = value
    if synonyms:
        attributes["Synonyms"] = synonyms

    return _ok(
        {
            "query_identifier": identifier,
            "query_source": source,
            "organism": organism,
            "attributes": attributes,
        },
        metadata={"source": "BridgeDb", "system_code": _bridgedb_code(source)},
    )


# --------------------------------------------------------------------------------------------
# HGNC -- the human gene naming authority
# --------------------------------------------------------------------------------------------


def _hgnc_fetch(field, value):
    """``(response_object, None)`` or ``(None, error_payload)`` for ``/fetch/<field>/<value>``.

    ``fetch`` is an exact, case-insensitive match on a stored field and returns complete gene
    records -- unlike ``search``, which is a scored Solr query returning only stubs.
    """
    payload, failure = _fetch_json(f"{_HGNC_BASE}/fetch/{_seg(field)}/{_seg(value)}")
    if failure:
        return None, failure
    return (payload or {}).get("response", {}) or {}, None


def _resolve_retired_symbol(requested):
    """Resolve a symbol that is not a *current* HGNC symbol.

    ``fetch/symbol/<X>`` only matches genes whose approved symbol is exactly ``X``. Genes are
    routinely renamed (CYP2E to CYP2E1 in 2002, DIA4 to NQO1) and the retired string survives on the
    current record as ``prev_symbol``/``alias_symbol``. Looking ``X`` up in those fields gives the
    caller the same rich record a direct hit would have produced, with the substitution disclosed --
    never silently.
    """
    for field, article, relation in _HGNC_SYMBOL_FALLBACKS:
        response, failure = _hgnc_fetch(field, requested)
        if failure:
            return failure
        docs = response.get("docs", [])
        if not docs:
            continue

        if len(docs) > 1:
            candidates = [{"symbol": d.get("symbol"), "hgnc_id": d.get("hgnc_id"), "name": d.get("name")} for d in docs]
            listed = ", ".join(f"{c['symbol']} ({c['hgnc_id']})" for c in candidates)
            return _error(
                f"'{requested}' is not a current HGNC gene symbol, and it is ambiguous: it is the "
                f"{relation} of {len(candidates)} different genes: {listed}. Re-query "
                "hgnc_fetch_gene_by_symbol with the intended current symbol.",
                metadata={
                    "source": "HGNC",
                    "query_field": "symbol",
                    "query_value": requested,
                    "num_found": response.get("numFound", len(docs)),
                    "resolved_from": requested,
                    "resolution_relation": field,
                    "ambiguous_candidates": candidates,
                },
            )

        gene = docs[0]
        current = gene.get("symbol")
        return _ok(
            gene,
            metadata={
                "source": "HGNC",
                "query_field": "symbol",
                "query_value": requested,
                "num_found": 1,
                "resolved_from": requested,
                "resolved_symbol": current,
                "resolution_relation": field,
                "symbol_resolution_note": (
                    f"'{requested}' is not a current HGNC gene symbol. It is listed as {article} "
                    f"{relation} of '{current}', whose record is returned here."
                ),
            },
        )

    return _ok(
        {},
        metadata={
            "source": "HGNC",
            "query_field": "symbol",
            "query_value": requested,
            "num_found": 0,
            "no_results_note": (
                f"'{requested}' is not known to HGNC as a current symbol, a previous symbol, or an "
                "alias symbol. Check the spelling, or use hgnc_search_genes (which supports "
                "wildcards and free-text search) to find the gene."
            ),
        },
    )


def hgnc_fetch_gene_by_symbol(symbol):
    """Fetch the authoritative HGNC record for a gene symbol.

    The one lookup that distinguishes "this gene does not exist" from "this gene was renamed". A
    symbol that is no longer current is resolved through its ``prev_symbol`` and ``alias_symbol``
    entries and the substitution is reported in ``metadata`` -- so an old marker list from a 2011
    paper still resolves, and you are told that it did.

    Parameters
    ----------
    symbol (str, required): Gene symbol, e.g. "BRCA1", "GFAP", or a retired one such as "CYP2E".

    Returns
    -------
    dict: ``{"status": "success", "data": {...}, "metadata": {...}}`` -- the complete gene record,
        with ``metadata["symbol_resolution_note"]`` present when a retired symbol was resolved, or a
        ``status: error`` payload when the symbol is the previous/alias symbol of several genes.

    Examples
    --------
    - hgnc_fetch_gene_by_symbol("BRCA1")
    - hgnc_fetch_gene_by_symbol("CYP2E")
    """
    if not symbol:
        return _missing("symbol", "Pass a gene symbol, e.g. 'BRCA1'.")
    response, failure = _hgnc_fetch("symbol", symbol)
    if failure:
        return failure
    docs = response.get("docs", [])
    if docs:
        return _ok(
            docs[0],
            metadata={
                "source": "HGNC",
                "query_field": "symbol",
                "query_value": symbol,
                "num_found": response.get("numFound", len(docs)),
            },
        )
    return _resolve_retired_symbol(symbol)


def hgnc_fetch_gene_by_id(hgnc_id):
    """Fetch the authoritative HGNC record for an HGNC ID.

    Parameters
    ----------
    hgnc_id (str, required): HGNC ID, with or without the prefix -- "HGNC:1100" and "1100" both
        work.

    Returns
    -------
    dict: ``{"status": "success", "data": {...}, "metadata": {...}}``, or a success payload with an
        empty ``data`` and a ``metadata["no_results_note"]`` when the ID is withdrawn or malformed.

    Examples
    --------
    - hgnc_fetch_gene_by_id("HGNC:1100")
    - hgnc_fetch_gene_by_id("1100")
    """
    if not hgnc_id:
        return _missing("hgnc_id", "Pass an HGNC ID such as 'HGNC:1100'.")
    value = str(hgnc_id)
    if not value.startswith("HGNC:"):
        value = f"HGNC:{value}"
    response, failure = _hgnc_fetch("hgnc_id", value)
    if failure:
        return failure
    docs = response.get("docs", [])
    if docs:
        return _ok(
            docs[0],
            metadata={
                "source": "HGNC",
                "query_field": "hgnc_id",
                "query_value": value,
                "num_found": response.get("numFound", len(docs)),
            },
        )
    return _ok(
        {},
        metadata={
            "source": "HGNC",
            "query_field": "hgnc_id",
            "query_value": value,
            "num_found": 0,
            "no_results_note": (
                f"No HGNC record has hgnc_id '{value}'. The ID may be withdrawn or malformed. Use "
                "hgnc_search_genes to look the gene up by symbol or name instead."
            ),
        },
    )


def _hgnc_search(query, search_field, location_echo=None):
    """Shared body of the two HGNC search tools."""
    if search_field:
        url = f"{_HGNC_BASE}/search/{_seg(search_field)}/{_seg(query)}"
    else:
        url = f"{_HGNC_BASE}/search/{_seg(query)}"
    payload, failure = _fetch_json(url)
    if failure:
        return failure

    response = (payload or {}).get("response", {}) or {}
    docs = response.get("docs", [])
    metadata = {
        "source": "HGNC",
        "total_results": response.get("numFound", len(docs)),
        "query": query,
        "query_location": location_echo if location_echo is not None else query,
    }
    if not docs:
        if search_field == "location":
            metadata["no_results_note"] = (
                f"No HGNC genes are mapped to location '{query}'. Locations are cytogenetic bands "
                "such as '17p13.1'; a broader band (e.g. '17p13') may return results."
            )
        else:
            metadata["no_results_note"] = (
                f"No HGNC genes matched '{query}'"
                + (f" in field '{search_field}'" if search_field else "")
                + ". Try a wildcard (e.g. 'BRCA*'), drop the search_field to search symbol and name "
                "together, or search 'prev_symbol'/'alias_symbol' for a retired name."
            )
    return _ok(docs, metadata=metadata)


def hgnc_search_genes(query, search_field=None):
    """Search HGNC for genes by symbol, name, alias or any indexed field.

    Wildcards work ("BRCA*"), which makes this the tool for enumerating a gene family by naming
    convention. Results are stubs (symbol, HGNC ID, name, score); pass a hit's symbol to
    ``hgnc_fetch_gene_by_symbol`` for the complete record.

    Parameters
    ----------
    query (str, required): Search term, wildcards allowed, e.g. "BRCA*", "collagen", "CD8A".
    search_field (str, optional): Restrict to one field, e.g. "symbol", "name", "alias_symbol",
        "prev_symbol". Omit to search symbol and name together.

    Returns
    -------
    dict: ``{"status": "success", "data": [...], "metadata": {"total_results": N, ...}}`` -- a list
        of stub records -- or a ``status: error`` payload. A miss carries
        ``metadata["no_results_note"]`` explaining what to try instead.

    Examples
    --------
    - hgnc_search_genes("BRCA*")
    - hgnc_search_genes("p65", search_field="alias_symbol")
    """
    if not query:
        return _missing("query", "Pass a search term, e.g. 'BRCA*' or 'collagen'.")
    return _hgnc_search(query, search_field)


def hgnc_search_by_location(location):
    """Find every HGNC gene mapped to a cytogenetic band.

    Parameters
    ----------
    location (str, required): Cytogenetic band, e.g. "17p13.1". A broader band ("17p13") matches
        more genes.

    Returns
    -------
    dict: ``{"status": "success", "data": [...], "metadata": {"total_results": N, ...}}`` or a
        ``status: error`` payload.

    Examples
    --------
    - hgnc_search_by_location("17p13.1")
    - hgnc_search_by_location("Xq28")
    """
    if not location:
        return _missing("location", "Pass a cytogenetic band, e.g. '17p13.1'.")
    return _hgnc_search(location, "location", location_echo=location)


def hgnc_fetch_gene_family_members(gene_group_id):
    """List every gene HGNC assigns to one gene family/group.

    Gene groups are HGNC's curated families -- "solute carriers", "collagens", "CD molecules". The
    ``gene_group_id`` of any gene appears in the record returned by ``hgnc_fetch_gene_by_symbol``,
    which is how you find the ID to pass here.

    Parameters
    ----------
    gene_group_id (str, required): HGNC gene group ID, e.g. "2155" (solute carrier family 2).

    Returns
    -------
    dict: ``{"status": "success", "data": [...], "metadata": {"num_found": N, ...}}`` -- one
        complete record per member gene -- or a ``status: error`` payload. An unknown group returns
        an empty list with ``metadata["no_results_note"]``.

    Examples
    --------
    - hgnc_fetch_gene_family_members("2155")
    - hgnc_fetch_gene_family_members("2247")
    """
    if gene_group_id is None or str(gene_group_id).strip() == "":
        return _missing(
            "gene_group_id",
            "Pass an HGNC gene group ID, e.g. '2155'. The 'gene_group_id' field of any record from "
            "hgnc_fetch_gene_by_symbol names the groups that gene belongs to.",
        )
    value = str(gene_group_id).strip()
    response, failure = _hgnc_fetch("gene_group_id", value)
    if failure:
        return failure
    docs = response.get("docs", [])
    metadata = {
        "source": "HGNC",
        "query_field": "gene_group_id",
        "query_value": value,
        "num_found": response.get("numFound", len(docs)),
    }
    if not docs:
        metadata["no_results_note"] = (
            f"HGNC has no gene family/group with gene_group_id '{value}'. Find a valid "
            "gene_group_id in the 'gene_group_id' field of any record returned by "
            "hgnc_fetch_gene_by_symbol."
        )
    return _ok(docs, metadata=metadata)
