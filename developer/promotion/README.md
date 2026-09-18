# Promotion Workflow

Promotion turns private, verified experiments into signed public knowledge. It
does not search for kernels, and the publisher never edits a reviewed public
dispatch winner.

## 1. Create a proposal

Run `create_proposal.py` as a module with the candidate pack, public dispatch
entry, correctness evidence, internal timing, independent comparison, pack
verification, and runtime verification. Store the output under the private lab
workspace. The proposal contains hashes and sanitized evidence, not source
paths or host identities.

```bash
uv run python -m developer.promotion.create_proposal \
  --pack <private-pack> \
  --dispatch dispatch/gemm-1.0.json \
  --entry-id <dispatch-entry-id> \
  --oracle <private-oracle.json> \
  --timing <private-timing.json> \
  --external <private-external-validation.json> \
  --pack-verification <private-pack-verification.json> \
  --runtime-verification <private-runtime-verification.json> \
  --output <private-proposal.json>
```

For generic LLM operator evaluations produced by
`benchmarks/llm_ops/evaluate.py`, use `prepare_llm_candidates.py` to create exact
dispatch entries and candidate packs, then `create_operator_proposal.py` to
create the privacy-reviewed proposal. It applies the same correctness, CV,
improvement and bootstrap confidence gates without assuming GEMM shapes or
`torch.mm`.

When one operator evaluation contains multiple promotable workload results, use
`prepare_operator_candidates.py`. It creates one immutable pack per exact
workload and a private staged dispatch containing every passing entry, while
leaving the public fallback-only dispatch unchanged. Create one
`create_operator_proposal.py` proposal per staged entry.

## 2. Record the human decision

The reviewer reads the proposal and records an explicit decision. The approval
is bound to the proposal file's exact SHA-256, so later edits invalidate it.

```bash
uv run python -m developer.promotion.record_approval \
  --proposal <private-proposal.json> \
  --reviewer <reviewer-id> \
  --approve \
  --reason "reviewed correctness, stability, scope, and lineage" \
  --output <private-approval.json>
```

`needs_review` proposals cannot be approved without `--override-gates`. An
overridden approval still cannot publish unless the publisher also receives
`--allow-override`. This two-step exception is for an auditable emergency or
preview decision, not routine promotion.

## 3. Publish

The publisher rechecks the proposal hash, approval, candidate manifest and
artifacts, current dispatch content, trusted signing key, and every signature it
creates. It refuses to overwrite immutable evidence, approvals, or archives.

```bash
uv run python -m developer.promotion.publish_approved \
  --proposal <private-proposal.json> \
  --approval <private-approval.json> \
  --pack <private-pack> \
  --dispatch dispatch/gemm-1.0.json \
  --private-key ~/.local/share/radeon-kernels/keys/<key-id>.private.json \
  --trusted-keys keys/trusted-keys.json \
  --public-root . \
  --release-root <private-release-output>
```

The resulting archive contains only the manifest, its detached signature, and
the artifacts named by the manifest. Raw evidence and machine-specific files
remain private. Publication is complete only after the signed archive passes
the target runtime verification.

During active development, target acceptance may synchronize the source tree
and public resources directly instead of rebuilding a wheel for every
promotion. `run_gemv_acceptance.sh` exercises this source mode while still
installing releases only through the signed Pack Index and disabling runtime
compilation and search. Build and test a wheel at milestone freezes and public
release boundaries, where the distributable itself is the object under test.

For operators with many exact workload winners, use
`create_operator_proposals.py`, `record_operator_approvals.py`, and
`publish_operator_approvals.py`. Each batch tool preflights the complete input
set and refuses overwrite or gate bypass; the underlying per-proposal publisher
still verifies every proposal hash, approval, dispatch entry, artifact, schema,
signature, and release archive independently.
