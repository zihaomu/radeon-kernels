# Radeon Kernels Architecture

Status: APPROVED

## Objective

`radeon-kernels` finds and publishes the fastest correct implementation of each
operator for each supported Radeon workload class. The multi-machine lab is an
internal execution facility, not the product.

The selection key is intentionally more precise than just an operator name:

```text
operator + Radeon architecture + dtype + layout + shape region + toolchain
```

The public result is an architecture-aware dispatch manifest backed by sanitized
benchmark evidence. Hostnames, SSH details, remote paths, registry locations and
raw lab results remain private to the local WSL workspace.

## Terms And Initial Scope

- **target**: one machine entry in the private `targets.yaml` file;
- **task**: one target, operator workload and candidate configuration combination;
- **attempt**: one execution of a task, including failed or retried executions;
- **candidate**: a named operator implementation with declared compatibility;
- **workload class**: dtype, layout and a bounded shape region;
- **current winner**: the dispatch entry currently selected for a complete
  operator, architecture, workload and toolchain key;
- **promotion**: generation of a reviewed dispatch proposal from benchmark evidence;
- **publication**: committing sanitized evidence and an approved dispatch manifest.

The first operator is GEMM. The first local slice supports configuration and
synthetic samples for FP16 and BF16, NN/NT/TN layouts and explicitly bounded M/N/K
regions. Real GPU architectures, Triton kernels, remote execution and performance
claims are non-goals until the local contracts pass their milestone tests.

## Workspace Boundary

The WSL workspace contains a public Git repository and private lab state as
siblings:

```text
radeon-kernel-workspace/
|-- .radeon-workspace.yaml
|-- radeon-kernels/                 # public Git repository
`-- lab-private/                    # never added to the public repository
    |-- config/targets.yaml
    |-- state/
    |-- runs/
    |-- logs/
    |-- cache/
    `-- exports/
```

The CLI locates `.radeon-workspace.yaml` by walking from the current directory
towards the filesystem root. Explicit `--workspace` always wins.

## Public Repository

```text
radeon-kernels/
|-- src/radeon_kernels/
|   |-- ops/
|   |   `-- gemm/
|   |       |-- api.py
|   |       |-- spec.py
|   |       |-- reference.py
|   |       `-- candidates/
|   |-- search/
|   |   `-- ranking.py
|   |-- lab/
|   |   |-- cli.py
|   |   |-- config.py
|   |   |-- planner.py
|   |   |-- remote.py
|   |   `-- results.py
|   `-- runtime/
|       |-- arch.py
|       |-- registry.py
|       |-- dispatcher.py
|       `-- compatibility.py
|-- dispatch/
|-- evidence/
|-- lab/
|   |-- targets.example.yaml
|   `-- target-schema.json
|-- environments/
|-- tests/
`-- docs/
```

Only directories needed by the first usable slice are created initially. Later
operator, dispatch and evidence directories are added with their first artifact.

## Private Target Configuration

There is one source of truth:

```text
lab-private/config/targets.yaml
```

Every target listed in the `targets` array is planned by default. A target omitted
from the file is invisible to the lab. There are no pools, priorities, roles or
`enabled` flags.

Each target defines:

- a stable local identifier and an SSH alias from `~/.ssh/config`;
- an optional expected GPU architecture, or `auto` for remote discovery;
- a remote workspace root owned by the SSH user;
- GPU IDs that the lab is allowed to use;
- per-target concurrency and busy-GPU behavior;
- an immutable container image digest and structured target-specific container
  options.

The target schema rejects unknown fields so misspelled configuration cannot be
silently ignored.

Secrets are never stored in the workspace. SSH keys remain in the user's SSH
configuration and registry credentials remain in the container runtime's
credential store.

SSH host fingerprints are stored separately in `lab-private/state/known_hosts`.
Unknown keys are rejected by default; the user must explicitly request first-seen
key acceptance. A changed fingerprint always fails and requires manual review.

Container configuration exposes structured device, mount, security-option and
host-environment-reference fields. Arbitrary shell fragments and literal secret
environment values are invalid. Registry and other credentials may only be
referenced by name and resolved by the remote runtime.

## Target Selection Semantics

Default planning is deliberately simple:

```text
load targets.yaml -> validate every entry -> plan every listed target
```

Every planned task is attempted according to its target's busy policy. `--target`
may select a subset, but only from entries already present in the private file.
When multiple configured targets report the same detected architecture, every one
remains a separate target and attempt. Results retain both `target_id` and the
detected architecture.

When `expected_architecture` is `auto`, the detected architecture becomes part of
the result. When a concrete expectation is configured, a mismatch fails preflight
before staging or container launch.

Targets contribute to one promotion only when the complete selection key and
environment compatibility key agree. The compatibility key includes the detected
architecture, driver, runtime, Triton version and image digest. Compatible targets
must select the same candidate. On every compatible target, the lower bound of a
95% paired bootstrap confidence interval for improvement over the current winner
must exceed the configured minimum improvement. Otherwise, the run is marked
`NEEDS_REVIEW`; the system does not choose a winner automatically. Human approval
requires a versioned override record naming the excluded evidence and rationale.

## GPU Allocation

`gpu_ids` is a static allowlist, not a statement that a GPU is currently idle.
Before each task, the remote executor probes processes, memory use, temperature
and clocks. `busy_policy` controls the response:

- `wait`: wait up to `busy_timeout_seconds`;
- `skip`: record a skipped task and continue the matrix;
- `fail`: fail the target immediately.

Each selected GPU also uses an atomic remote `flock` at the canonical per-user path
`${XDG_RUNTIME_DIR:-/tmp}/radeon-kernels-$UID/locks/gpu-<id>.lock`. This path is not
configurable, so cooperating controllers cannot accidentally choose different lock
names. The lock process holds the file descriptor for the full attempt and writes
owner, run ID, attempt ID and start time to adjacent metadata.

Each physical target must be managed through one configured SSH alias and Unix
account. Configuration validation rejects duplicate physical-target identities;
using multiple Unix accounts for one GPU host is unsupported because per-user locks
cannot coordinate them.

The remote lock-holder starts the attempt under a remote watchdog with a hard
attempt deadline. If the controller disconnects, the watchdog continues locally,
terminates the process tree at the deadline and then exits, releasing the kernel
lock. The next preflight removes metadata only after proving that no process holds
the lock. The lock coordinates cooperating controllers only. Process and telemetry
checks detect non-cooperating workloads.

The executor acquires the cooperative lock before checking processes, memory use,
temperature and clocks. If the GPU is busy, it releases or waits according to the
configured policy. This ordering closes the gap between probing and acquisition.

## Execution Lifecycle

Every task follows the same state machine:

```text
validate -> preflight -> acquire GPU lock -> stage source -> start container
-> correctness gate -> start telemetry sampler -> warmup -> timed trials
-> stop and finalize telemetry
-> validate result -> return artifacts -> cleanup -> release lock
```

Task status is one of `PLANNED`, `RUNNING`, `SUCCEEDED`, `SKIPPED`, `FAILED` or
`CANCELLED`. A run aggregates to `SUCCEEDED`, `PARTIAL`, `FAILED` or
`NEEDS_REVIEW`. Cleanup failure is recorded independently and changes an otherwise
successful task to `FAILED`. Every status transition is appended to the private
manifest with a timestamp and reason.

Run aggregation is deterministic: any unresolved evidence conflict yields
`NEEDS_REVIEW`; all successful tasks yields `SUCCEEDED`; no successful tasks and at
least one failure or cancellation yields `FAILED`; no successful tasks and all
tasks skipped yields `PARTIAL`; every other mixture of success, failure, skip or
cancellation yields `PARTIAL`.

SSH connect, staging, image preparation, correctness, benchmark, collection and
cleanup each have explicit configurable timeouts. Cancellation stops new tasks,
terminates the active remote process, collects whatever logs exist and releases the
lock. One failed target never blocks completion of unrelated targets.

Infrastructure failures may be retried. Timed trials are never retried and then
selectively reduced to the best result. A retry creates a new attempt and retains
the failed attempt for auditability.

The default limit is two attempts with exponential backoff starting at five
seconds. SSH connect, staging, image retrieval and artifact collection failures are
retryable. Configuration, architecture mismatch, correctness, benchmark process,
busy-policy, cancellation and result-validation failures are not. A task succeeds
when any attempt succeeds, but every attempt remains recorded. Exhausted retryable
failures produce `FAILED`; a busy-policy skip produces `SKIPPED` without retry.

## Search And Promotion

Candidate implementations declare supported architectures, dtypes, layouts and
shape constraints. Search spaces declare tuning parameters separately from the
kernel source.

A candidate can be promoted only when:

1. It passes correctness against a reference implementation.
2. It is measured in randomized order against the current winner.
3. All raw samples are retained.
4. Runtime telemetry is within configured stability bounds.
5. The improvement exceeds both a minimum percentage and the measured noise.
6. All compatible targets satisfy the paired confidence rule, or a human records
   an explicit review override.
7. The evidence is sanitized before entering the public repository.

Every workload definition must provide deterministic input seeds, dtype-specific
absolute and relative tolerances, warmup count, sample count, stability threshold
and minimum improvement. Initial defaults are 10 warmups, 30 measured samples,
coefficient of variation at most 3% and minimum median improvement of 3%. Raw
samples are never deleted as outliers. Ranking uses the median and a bootstrap
confidence interval; unstable data yields `NEEDS_REVIEW`. Temperature, clocks and
memory use are sampled before, during and after timed trials.

Correctness specifications also define input distributions and ranges, reference
accumulation precision, boundary shapes, and expected NaN/Inf behavior. The initial
GEMM suite uses deterministic uniform and normal finite inputs plus explicit zero,
unit, non-multiple tile and configured maximum boundary shapes. Non-finite inputs
are tested only by workloads whose contract defines their behavior.

Promotion generates a proposed dispatch change. It never commits or publishes the
change automatically.

## Dispatch Contract

Dispatch manifests are JSON documents with a required string `schema_version` in
`MAJOR.MINOR` form, initially `1.0`. Each entry contains the complete selection key,
candidate content hash, tuning parameters, toolchain compatibility range and
evidence ID. Shape regions
must not overlap at the same priority. The validator rejects ambiguous regions;
more-specific dispatch is represented with an explicit higher integer priority,
and its region must be a strict subset of every lower-priority region it overlaps.
Arbitrary partial shadowing is invalid.

Each operator dispatch document declares a top-level `fallback_candidate_id`.
Fallback is resolved from the candidate registry and validated against architecture,
dtype, layout and toolchain constraints when the manifest is loaded.

At runtime, entries are filtered by exact operator, architecture, dtype and layout,
then by toolchain compatibility and shape containment, and finally sorted by
priority. No match uses the declared portable fallback. No compatible fallback is
a clear unsupported-workload error, never an implicit arbitrary candidate.

Dispatch schema changes are backward compatible within a major version. Additive
changes increment `MINOR`; removing or changing field meaning increments `MAJOR`
and requires an explicit migration tool.

## Result Boundaries

Raw, private run data is stored below:

```text
lab-private/runs/<run-id>/
|-- manifest.json
|-- plan.json
|-- targets/<target-id>/environment.json
|-- targets/<target-id>/trials.jsonl
|-- targets/<target-id>/stdout.log
`-- aggregate/summary.json
```

Public evidence excludes SSH aliases, IP addresses, user names, absolute paths,
registry credentials and container registry locations. Sanitization is implemented
as a public-field allowlist rather than a forbidden-field blocklist. The allowlist
retains schema version, source commit, candidate content hash, image content digest,
operator, workload, detected GPU properties, public driver/runtime/Triton versions,
candidate ID, configuration, sanitized timing samples and statistical summary.
Private raw artifacts retain the full image reference and all unmodified samples;
public evidence contains only the allowlisted subset.

The public GPU-property allowlist is limited to architecture name, marketing model,
compute-unit count, wavefront size, memory capacity and declared feature flags.
Serial numbers, device UUIDs, PCI addresses, topology paths and host-local device
indices are always excluded.

## Initial Slice

The first usable slice contains:

1. Python project metadata and an `rk` CLI.
2. Workspace discovery and strict target configuration validation.
3. A planner whose default matrix contains every configured target.
4. A `doctor` command that can inspect configuration without contacting targets.
5. A versioned result model and sanitizer.
6. A GEMM workload model and deterministic winner-ranking logic.
7. Unit tests using fake targets and synthetic benchmark samples.

Remote execution and real Triton kernels are added after this local control-plane
contract is stable. This keeps the first implementation verifiable on WSL without
requiring PyTorch, Triton or a GPU.

Milestone exits are:

1. **Local contract**: workspace, target, result, sanitization and ranking tests pass.
2. **Remote protocol**: fake-SSH and one opt-in target complete all failure paths.
3. **Real GEMM**: correctness and benchmark methodology pass on one architecture.
4. **Multi-target verification**: compatible duplicate targets agree or emit a
   review record.
5. **Publication**: dispatch validation and public-evidence leak tests pass before
   any performance claim is committed.

## Success Criteria

- Private targets can be added or removed without modifying the public repository.
- Default planning includes exactly the targets listed in `targets.yaml`.
- An unknown target requested by CLI is rejected before any SSH command.
- Duplicate architectures remain separate in raw results.
- Configuration mistakes fail with a precise field path.
- Synthetic stable samples promote a clear winner.
- Noisy or conflicting samples produce `NEEDS_REVIEW`.
- Sanitized evidence contains no machine identity or absolute path.
