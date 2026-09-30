# Dataset distribution

Current local contents:

- `2D/train/image` and `2D/train/label`: 10,735 PNG/polygon-TXT pairs.
- `2D/test/image` and `2D/test/label`: 2,945 PNG/polygon-TXT pairs.
- `3D/DAPI_Tag_matched_1333pairs.tar.gz`: existing paired DAPI/Tag archive.

The 2D inventory excludes robustness, misc, and visualization assets. Its
logical size is 4,890,159,160 bytes (4.89 GB / 4.55 GiB). Stratified sampling
estimates gzip level-6 tar archives at about 2.79 GB for train and 0.81 GB for
test, or **3.61 GB total**. Plan for 3.42–3.79 GB; this is an engineering
allowance, not a statistical confidence interval or a measured full archive.

The existing 3D archive is **37,556,932,777 bytes** (37.56 GB / 34.98 GiB),
so no additional compression estimate is needed. It is a DAPI/Tag package,
not a claim that separately exported GT instance masks are included. Estimated
combined 2D/3D compressed storage is 41.16 GB, excluding weights and future 4D
data. The release currently does not include the complete Chimps training set
or historical ChromoLift training features; see the model READMEs.

## Recheck the estimate

```bash
python Dataset/estimate_archive_size.py --sample-mib 128 \
  --output Dataset/archive_size_estimate.json
```

The report records exact counts/bytes and deterministic sampling by split and
extension. The estimator does not create full archives or modify source files.

## Publication recommendation

Keep source code and the approximately 40 MB of UMAP artifacts on GitHub.
Use separate Zenodo records for the versioned 2D and 3D paper datasets, each
with a DOI; this leaves room for later 4D releases and independent revisions.
Both current packages could also fit a single record: Zenodo's default limit
is 50 GB total and 100 files per record. See
[Zenodo file limits](https://help.zenodo.org/docs/deposit/manage-files/).

Hugging Face dataset repositories are a convenient working distribution or
download mirror. The current free account has 100 GB of private storage across
repositories; public storage is best-effort, not an unlimited guaranteed quota.
Check account usage before uploading. See
[HF storage limits](https://huggingface.co/docs/hub/storage-limits).

Personal Figshare currently advertises 20 GB private storage and a 20 GB
single-file limit, so the existing 37.56 GB 3D archive is not suitable as one
file there without an institutional/custom arrangement. See
[Figshare upload guidance](https://help.figshare.com/article/how-to-upload-and-publish-your-research).

Before publishing, verify archive integrity, document the extracted layout,
add SHA-256 checksums and an approved data license, and supply download URLs
here. No dataset has been uploaded by this reorganization.
