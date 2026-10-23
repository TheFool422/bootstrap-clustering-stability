# First-payload traffic clustering

This package compares five session representations: metadata (**H**), first client payload (**F**), retained client payload sequence (**A**), and their concatenations (**HF**, **HA**). It contains three executable scripts and the saved inputs and scores used in the experiments. All commands below run from this directory.

## Files

| File | Role |
|---|---|
| `extract.pl` | Extract TCP sessions, reconstruct payloads, match benchmark labels, and select traffic samples. |
| `sessions.zip` | Eight extracted CSVs: W1–W4 from each dataset, 10,000 sessions each. |
| `embed.py` | Build H and generate byte-mean BigBird or BERT embeddings for F and A. |
| `cluster.py` | Dimensionality reduction, baseline clustering, column-bootstrap clustering, stability scores, and summary reports. |
| `results.csv` | Archived per-configuration scores for P0–P4 at all five thresholds. These are historical results, not outputs from the compact scripts. |
| `settings.json` | Grids, seeds, extraction settings, pinned model revisions, recorded environment and embedding times, and source-file checksums. |
| `requirements.txt` | Recorded Python library versions, excluding PyTorch. |
| `README.md` | File descriptions, procedures, and commands. |
| `.gitignore` | Keeps archives, downloaded models, and newly generated files out of ordinary Git commits. |

The extracted data in `sessions.zip` are provided, but **`embeddings.zip` is not uploaded or distributed because of its large size**. Users who wish to rerun the experiments must generate the embeddings themselves using `embed.py`, as described in Step 2. The saved `results.csv` can be inspected without generating embeddings.

All executable code is visible in the three scripts. Raw PCAPs, model weights, manuscript files, personal paths, and original Git history are not included.

## Start with the saved results

Python 3.10 or later is sufficient for this command; no GPU or scientific Python packages are needed:

```console
python cluster.py summarize
```

`summarize` prints mean scores as percentages and separates F/A wins, ties, and losses. It does not launch new experiments. To save the report:

```console
python cluster.py summarize > summary.csv
```

The report has two CSV sections: means, followed by matched F/A comparison counts.

## Environment

The recorded environment is Windows with Python 3.10.11. Install the recorded libraries in a separate environment:

```console
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

For embedding, also install a CUDA-compatible PyTorch build. The recorded PyTorch/CUDA version is in `settings.json`; select a build compatible with the available GPU and driver. Embedding requires a CUDA GPU. Extraction requires Perl and Wireshark's `tshark` and `editcap` on PATH. Model weights are downloaded from pinned Hugging Face revisions when `embed.py` first runs. The existing installed environment was used for package checks; a fresh installation has not been tested.

## 1. Extract sessions (optional)

Skip this step when using `sessions.zip`. To extract again, obtain the original benchmark captures and label files and arrange them under a raw-data directory:

```text
raw_data/
  UNSW-NB15/
    1.pcap, ..., 8.pcap
    UNSW-NB15_1.csv, ..., UNSW-NB15_4.csv
  CICIDS2017/
    Tuesday-WorkingHours.pcap
    TrafficLabelling/Tuesday-WorkingHours.pcap_ISCX.csv
```

```console
perl extract.pl --dataset unsw_nb15 --sample W1 --raw raw_data
perl extract.pl --dataset cicids2017 --sample W2 --raw raw_data
```

Change `--sample` to W1, W2, W3, or W4 and repeat for both datasets as needed. Paths containing spaces must be quoted. Optional `--tshark` and `--editcap` arguments accept executable paths.

Outputs go under `generated/extraction/<dataset>/<sample>/`. The script refuses to reuse an existing sample output directory; use another `--output` directory for a new run. Window slicing and intermediate files are written there, leaving source PCAPs unchanged.

The supplied settings preserve the historical extraction procedure: complete captured handshake, nonempty client payload, session ordering by SYN time, and exclusion of ambiguous or unmatched benchmark labels during sample selection. Labels are not clustering features or stability targets. W2–W4 use hour offsets from the first captured packet, as in the original extraction scripts. F uses the first retained client segment with the recorded MSS/timestamp clipping rule; A reconstructs client bytes in sequence order and retains up to 4,094 bytes. Token limits are then applied during embedding. The saved CSVs retain extraction diagnostics and labels for inspection.

## 2. Generate embeddings

This step is required before rerunning clustering because the embedding archive is not distributed. Generate BigBird embeddings first, then BERT embeddings for the same W1 sample; BERT uses the retained-byte indexes produced by the BigBird run.

```console
python embed.py --dataset unsw_nb15 --sample W1 --model bigbird
python embed.py --dataset unsw_nb15 --sample W1 --model bert
```

Repeat BigBird for W1–W4 of both datasets, and BERT for W1 of both datasets. Default input is `sessions.zip`; it does not need to be unpacked. To use a newly extracted CSV:

```console
python embed.py --dataset unsw_nb15 --sample W1 --model bigbird --sessions generated/extraction/unsw_nb15/W1/sessions_first10k.csv
```

Results are written to `generated/embeddings/<dataset>/<sample>/`: H metadata, F/A arrays, retained-byte indexes, and timing records. A new run must use a fresh `--output` directory if the requested embeddings already exist. `--only F` or `--only A` selects one payload representation. `--checkpoint` uses a local model instead of downloading the pinned revision.

Each byte is represented as a two-digit hexadecimal word. BigBird retains whole-byte prefixes within 4,094 content tokens, uses separate full/sparse attention batches, and averages token vectors within each byte before averaging bytes. BERT uses the same retained bytes as BigBird, splits them into whole-byte chunks of at most 510 content tokens, and combines chunk embeddings by byte count. Thus, BERT does not simply discard everything after its first 512-token input. Saved embeddings are float32, with one row per session in CSV order.

## 3. Reduce, cluster, and evaluate stability

After Step 2, start with a small execution check using your generated embeddings:

```console
python cluster.py run --phases P0 --dataset unsw_nb15 --representations H F --dimensions 5 --cluster-sizes 13 --replicas 3 --workers 1 --embeddings generated/embeddings --output generated/check
```

This is a small check, not a reproduction of the published scores. After generating BigBird embeddings for all eight samples and BERT embeddings for W1 of both datasets, run the full configured experiments:

```console
python cluster.py run --phases P0 P1 P2 P3 P4 --workers 6 --embeddings generated/embeddings --output generated/full
python cluster.py summarize --results generated/full/results.csv
```

Full bootstrap evaluation is computationally expensive. `--workers` controls concurrent jobs; use fewer workers if memory is limited. Always pass `--embeddings generated/embeddings` (or your chosen output directory), because the script's legacy default refers to the undistributed archive. To run only P0 for one dataset:

```console
python cluster.py run --phases P0 --dataset unsw_nb15 --embeddings generated/embeddings --output generated/new_embeddings
```

| Phase | Samples | Embedding | Reduction | Clustering |
|---|---|---|---|---|
| P0 | W1 | BigBird | UMAP | HDBSCAN |
| P1 | W2–W4 | BigBird | UMAP | HDBSCAN |
| P2 | W1 | BERT | UMAP | HDBSCAN |
| P3 | W1 | BigBird | PCA | HDBSCAN |
| P4 | W1 | BigBird | UMAP | DBSCAN |

P0 uses d = 5, 10, ..., 50, 60, 80, 100 and c = 5, 7, ..., 21, 30, 50, 100. P1–P4 use d = 5, 25, 50 and c = 5, 13, 50. H is evaluated only over clustering settings, without duplicating reduction dimensions. The phase names in this package follow the paper; the earlier scripts called the BERT/PCA/DBSCAN alternatives P1/P2/P3.

UMAP uses the recorded settings and seed 21. PCA uses full SVD without whitening; the smaller dimensions are prefixes of a 50-dimensional basis. The matrices are standardized before clustering. HDBSCAN uses `min_cluster_size=c` with its recorded defaults. For DBSCAN, c controls the neighbor order: the radius is the 0.75, 0.90, or 0.95 quantile of c-th-other-neighbor distances and `min_samples=c+1`. The radius-selection rule is reapplied to each replica. Fixed-radius sensitivity runs are excluded from the paper's P4 results.

Each configuration uses 1,000 column-bootstrap replicas. Replica b uses the original `RandomState(b)` seed, for b = 0,...,999. Embeddings, reduced coordinates, and session rows remain fixed. ARI treats noise labels as one group and is stored as float32 before thresholding, matching the recorded routine. S-scores are the fractions with ARI at least 0.5, 0.6, 0.7, 0.8, and 0.9.

The output directory retains one audit NPZ per representation/dimension job, including baseline labels, replica ARIs, and reduced features when applicable. Completed jobs with matching input hashes, settings, and library versions are reused. `results.csv` in that directory summarizes only the jobs requested in that invocation; rerun with all desired phases to assemble their combined CSV. The distributed historical `results.csv` is never overwritten.

## Payload and embedding cost

Each `embed.py` run writes retained-byte indexes and F/A timing JSON files under `generated/embeddings/<dataset>/<sample>/<model>/`. These report the average retained bytes and embedding time per session for your run. Recorded BigBird times from the original experiments are also retained in `settings.json`. Historical byte averages use all four samples; timing averages use complete runs only. UNSW W1 is excluded from timing because it resumed cached chunks, and CICIDS W4 uses the earlier complete pass. New timings exclude model initialization and final output serialization; consolidation changes file-writing overhead, so they should not be presented as the historical measurements.

The `cluster.py verify` and `cluster.py costs` commands are retained for local auditing of the original archives. They require the undistributed `embeddings.zip` and are not steps in the public reproduction workflow.

## Checks and distribution

The compact package preserves the original data byte-for-byte and consolidates the archived scores without rerunning the historical experiments. The three scripts are a refactoring of the earlier implementation. Checks passed for all 53 archived file hashes and all 9,480 score rows. The extraction stages and their public command matched the earlier scripts on a packet fixture with retransmissions, out-of-order segments, and clipping. A selected 10,000-session UMAP output matched the saved coordinates exactly, and the checked H/F bootstrap ARIs matched the historical first three replicas. HDBSCAN/DBSCAN rules and pooling helpers matched their original routines. BigBird and BERT checks on 20 F sessions matched saved vectors within floating-point tolerance (minimum cosine similarity above 0.99999). A full end-to-end rerun of every configuration has not been performed for this package.

