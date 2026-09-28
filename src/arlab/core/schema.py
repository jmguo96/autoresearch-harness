"""Domain-neutral contracts. Algorithm-specific types live in task packs."""
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class ModelConfig(StrictModel):
    provider: Literal["demo", "compatible", "responses", "anthropic"] = "demo"
    model: str = "deterministic-teaching-policy"
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "ARLAB_API_KEY"
    max_output_tokens: int = Field(default=4096, ge=128, le=32768)
    timeout_seconds: float = Field(default=120, gt=0, le=600)
    retries: int = Field(default=2, ge=0, le=5)
    # Narrow owner-configured exception for the supplied lab origin only.
    allow_lab_http: bool = False
    # JSON schema support varies between compatible servers. Validation never becomes optional.
    json_mode: bool = True


class Budget(StrictModel):
    max_checks: int = Field(default=20, ge=0)
    max_check_ticks: int = Field(default=20, ge=1, le=100)
    max_decisions: int = Field(default=40, ge=1, le=100000)
    max_trials: int = Field(default=20, ge=1, le=100000)
    max_attempts: int = Field(default=100000, ge=1)
    max_attempts_per_trial: int = Field(default=3, ge=1, le=20)
    max_train_steps: int = Field(default=1_000_000, ge=1)
    max_optimizer_steps: int = Field(default=100_000_000, ge=1)
    max_eval_episodes: int = Field(default=200, ge=1)
    max_model_calls: int = Field(default=80, ge=0)
    max_output_tokens: int = Field(default=160000, ge=0)
    max_wall_seconds: float = Field(default=3600, gt=0)
    worker_timeout_seconds: float = Field(default=600, gt=0)


class ResearchAccess(StrictModel):
    """Owner grants outbound research access separately from candidate code execution."""
    enabled: bool = False
    web_search_enabled: bool = False  # Optional owner opt-in; not needed by the classroom.
    allowed_hosts: list[str] = Field(default_factory=lambda: [
        "export.arxiv.org", "arxiv.org", "api.crossref.org", "api.github.com",
        "raw.githubusercontent.com", "api.search.brave.com", "proceedings.mlr.press",
        "openreview.net", "docs.pytorch.org", "pytorch.org", "docs.github.com",
    ], max_length=100)
    max_requests: int = Field(default=100, ge=0, le=10000)
    max_download_bytes: int = Field(default=100_000_000, ge=1024)
    response_bytes: int = Field(default=8_000_000, ge=1024, le=30_000_000)
    timeout_seconds: float = Field(default=30, gt=0, le=120)
    retries: int = Field(default=1, ge=0, le=3)
    search_cache_seconds: int = Field(default=86400, ge=0)
    github_token_env: str = "ARLAB_GITHUB_TOKEN"
    brave_key_env: str = "BRAVE_SEARCH_API_KEY"
    user_agent: str = "AutoresearchLab/0.6 (bounded research client)"


class Case(StrictModel):
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,100}$")
    split: Literal["dev", "validation", "test"] = "dev"
    data: str
    environment: dict[str, Any]
    dataset_sha256: str = Field(default="", pattern=r"^([0-9a-f]{64})?$")


class CodeExecutionPolicy(StrictModel):
    """Owner-controlled, immutable permission; never supplied by a candidate.

    trusted_local executes reviewed Python with host-user privileges. It is NOT
    a security sandbox. docker runs candidate training and policy inference in
    separate restricted containers; the evaluator remains outside the policy
    container and never imports its code.
    """
    mode: Literal["disabled", "trusted_local", "docker"] = "disabled"
    docker_image: str | None = None
    max_parameters: int = Field(default=50_000_000, ge=1)
    max_artifact_bytes: int = Field(default=512_000_000, ge=1024)
    max_optimizer_steps_per_tick: int = Field(default=64, ge=1, le=256)
    inference_timeout_seconds: float = Field(default=15, gt=0, le=600)
    startup_timeout_seconds: float = Field(default=90, gt=0, le=600)
    container_memory: str = Field(default="8g", pattern=r"^[1-9][0-9]*[mg]$")
    container_cpus: float = Field(default=2, gt=0, le=256)
    container_gpu: bool = False

    @model_validator(mode="after")
    def image_required(self) -> CodeExecutionPolicy:
        if self.mode == "docker" and not self.docker_image:
            raise ValueError("docker code execution requires an immutable image name/digest")
        return self


class FinalEvaluation(StrictModel):
    """Predeclared owner-only confirmation panel; never a tool of the researcher.

    A case may also be a development case. Reports must disclose this overlap;
    fresh seeds do not make a previously used task a held-out task.
    """
    cases: list[str] = Field(min_length=1)
    train_seeds: list[int] = Field(default_factory=lambda: [41, 43, 47, 53, 59], min_length=1)
    eval_seeds: list[int] = Field(default_factory=lambda: list(range(9000, 9010)), min_length=1)
    steps: int = Field(default=1_000_000, gt=0)
    max_optimizer_steps: int = Field(default=8_000_000_000, gt=0)
    retry_reserve_fraction: float = Field(default=0.25, ge=0, le=2)
    max_attempts_per_trial: int = Field(default=3, ge=1, le=20)
    max_wall_seconds: float = Field(default=30 * 86400, gt=0)
    worker_timeout_seconds: float = Field(default=6 * 3600, gt=0)

    @model_validator(mode="after")
    def unique_panel(self):
        for values in (self.cases, self.train_seeds, self.eval_seeds):
            if len(set(values)) != len(values):
                raise ValueError("Final evaluation identities/seeds must be unique")
        if any(s < 0 or s > 2**31 - 1 for s in self.train_seeds + self.eval_seeds):
            raise ValueError("Final seeds must be nonnegative 31-bit integers")
        return self


class ExplorationPolicy(StrictModel):
    """Cheap partial measurements, selected by the model within a fixed dev pool.

    All scored checkpoints use one schedule horizon. The final development panel
    still uses all dev cases and confirmation_seeds at the longest horizon.
    """
    adaptive: bool = False
    minimum_steps: int = Field(default=1000, ge=1)
    initial_steps: int = Field(default=5000, ge=1)
    initial_eval_episodes: int = Field(default=3, ge=1)
    confirmation_seeds: list[int] | None = None


class ResearchContract(StrictModel):
    schema_version: int = 1
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    goal: str = Field(min_length=1, max_length=10000)
    task: str = "iql"
    cases: list[Case] = Field(min_length=1)
    seed_recipe: dict[str, Any]
    train_seeds: list[int] = Field(default_factory=lambda: [0, 1, 2], min_length=1)
    eval_seeds: list[int] = Field(default_factory=lambda: [1000, 1001, 1002], min_length=1)
    rungs: list[int] = Field(default_factory=lambda: [1000, 10000], min_length=1)
    budget: Budget = Field(default_factory=Budget)
    model: ModelConfig = Field(default_factory=ModelConfig)
    research_access: ResearchAccess = Field(default_factory=ResearchAccess)
    device: str = "cpu"
    code_execution: CodeExecutionPolicy = Field(default_factory=CodeExecutionPolicy)
    # Training is always offline. This flag records the source of research feedback.
    feedback: Literal["budgeted_simulator"] = "budgeted_simulator"
    # Active hypothesis branches (including baseline), NOT accumulated revisions.
    max_candidates: int = Field(default=8, ge=1, le=10000)
    max_revisions: int = Field(default=10000, ge=1, le=100000)
    exploration: ExplorationPolicy = Field(default_factory=ExplorationPolicy)
    final_checkpoint_only: Literal[True] = True
    final_evaluation: FinalEvaluation | None = None

    @model_validator(mode="after")
    def coherent(self) -> ResearchContract:
        names = [case.name for case in self.cases]
        if len(names) != len(set(names)) or not any(c.split == "dev" for c in self.cases):
            raise ValueError("Case names must be unique and at least one dev case is required")
        if self.rungs != sorted(set(self.rungs)) or self.rungs[0] <= 0:
            raise ValueError("Rungs must be strictly increasing positive update counts")
        policy = self.exploration
        if policy.adaptive:
            if not policy.minimum_steps <= policy.initial_steps <= self.rungs[-1]:
                raise ValueError("Adaptive initial steps must fit the fixed schedule horizon")
            if policy.initial_eval_episodes > len(self.eval_seeds):
                raise ValueError("Initial evaluation exceeds the authorized episode pool")
        if policy.confirmation_seeds is not None:
            if (not policy.confirmation_seeds or
                    len(set(policy.confirmation_seeds)) != len(policy.confirmation_seeds) or
                    not set(policy.confirmation_seeds).issubset(self.train_seeds)):
                raise ValueError("Confirmation seeds must be a nonempty subset of allowed training seeds")
        final = self.final_evaluation
        if final:
            if not set(final.cases).issubset(names):
                raise ValueError("Final evaluation references undeclared cases")
            if set(final.train_seeds) & set(self.train_seeds) or set(final.eval_seeds) & set(self.eval_seeds):
                raise ValueError("Final evaluation must use fresh training and rollout seeds")
            if final.steps < self.rungs[-1]:
                raise ValueError("Final training horizon cannot be below the longest search horizon")
        for seeds in (self.train_seeds, self.eval_seeds):
            if len(set(seeds)) != len(seeds) or any(s < 0 or s > 2**31-1 for s in seeds):
                raise ValueError("Seeds must be unique nonnegative 31-bit integers")
        return self


class Hypothesis(StrictModel):
    question: str = Field(min_length=1, max_length=4000)
    prediction: str = Field(min_length=1, max_length=4000)
    rationale: str = Field(min_length=1, max_length=4000)
    falsification: str = Field(min_length=1, max_length=4000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=30)


class PlanItem(StrictModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,50}$")
    objective: str = Field(min_length=1, max_length=1000)
    depends_on: list[str] = Field(default_factory=list)
    state: Literal["pending", "done", "blocked"] = "pending"


class PlanAction(StrictModel):
    kind: Literal["plan"]
    items: list[PlanItem] = Field(min_length=1, max_length=30)
    reason: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def acyclic(self) -> PlanAction:
        graph = {item.id: item.depends_on for item in self.items}
        if len(graph) != len(self.items):
            raise ValueError("Duplicate plan item")
        seen: set[str] = set()
        active: set[str] = set()
        def visit(node: str) -> None:
            if node not in graph or node in active:
                raise ValueError("Missing dependency or cycle")
            if node in seen:
                return
            active.add(node)
            for dep in graph[node]:
                visit(dep)
            active.remove(node)
            seen.add(node)
        for node in graph:
            visit(node)
        return self


class ProposeAction(StrictModel):
    kind: Literal["propose"]
    parent: str
    hypothesis: Hypothesis
    recipe: dict[str, Any]
    new_branch: bool = True
    review_id: str = ""


class ReviewProposalAction(StrictModel):
    """A separate, persisted self-check before any candidate is registered or edited."""
    kind: Literal["review_proposal"]
    parent: str
    hypothesis: Hypothesis
    method_change: str = Field(min_length=1, max_length=2000)
    why_not_tuning: str = Field(min_length=1, max_length=2000)
    feasibility: str = Field(min_length=1, max_length=2000)
    failure_mode: str = Field(min_length=1, max_length=2000)
    verdict: Literal["proceed", "reject"]


class SourceEdit(StrictModel):
    kind: Literal["write", "replace", "delete"]
    path: str = Field(min_length=1, max_length=200)
    old: str = Field(default="", max_length=1_000_000)
    new: str = Field(default="", max_length=1_000_000)


class EditAlgorithmAction(StrictModel):
    """Immutable source edit; never an edit to the running evaluator or harness."""
    kind: Literal["edit_algorithm"]
    parent: str
    new_branch: bool = False
    name: str = Field(min_length=1, max_length=120)
    hypothesis: Hypothesis
    review_id: str = ""
    template: str | None = None
    edits: list[SourceEdit] = Field(default_factory=list, max_length=64)
    config_updates: dict[str, Any] = Field(default_factory=dict)
    config_remove: list[str] = Field(default_factory=list)
    entrypoint: str | None = None
    max_optimizer_steps_per_tick: int | None = Field(default=None, ge=1, le=256)


class RunAction(StrictModel):
    kind: Literal["run"]
    candidate: str
    case: str
    train_seed: int
    steps: int = Field(gt=0)
    eval_episodes: int | None = Field(default=None, ge=1)


class ContinueTrialAction(StrictModel):
    kind: Literal["continue_trial"]
    trial: str
    steps: int = Field(gt=0, description="Cumulative target, not additional steps")
    eval_episodes: int | None = Field(default=None, ge=1)


class BranchStatusAction(StrictModel):
    kind: Literal["set_branch_status"]
    candidate: str
    active: bool
    reason: str = Field(min_length=1, max_length=2000)


class ReadTrialLogAction(StrictModel):
    kind: Literal["read_trial_log"]
    trial: str
    stream: Literal["train", "evaluation", "metrics", "train_error", "evaluation_error", "value_boundary"] = "train"
    start_line: int = Field(default=1, ge=1)
    max_lines: int = Field(default=60, ge=1, le=150)
    max_characters: int = Field(default=5000, ge=500, le=6000)
    offset: int = Field(default=0, ge=0)
    tail: bool = False
    start_step: int | None = Field(default=None, ge=0)
    end_step: int | None = Field(default=None, ge=0)
    fields: list[str] = Field(default_factory=list, max_length=20)


class InspectAction(StrictModel):
    kind: Literal["inspect"]
    target: Literal["candidate", "trial", "skill", "template", "schema", "plan", "feedback", "tool"]
    id: str = ""
    path: str = Field(default="", max_length=200)
    start_line: int = Field(default=1, ge=1)
    max_lines: int = Field(default=60, ge=1, le=150)
    max_characters: int = Field(default=5000, ge=500, le=6000)
    offset: int = Field(default=0, ge=0)


class SearchLiteratureAction(StrictModel):
    kind: Literal["search_literature"]
    query: str = Field(min_length=1, max_length=1500)
    provider: Literal["arxiv", "crossref"] = "arxiv"
    offset: int = Field(default=0, ge=0, le=10000)
    limit: int = Field(default=5, ge=1, le=20)
    refresh: bool = False


class SearchWebAction(StrictModel):
    kind: Literal["search_web"]
    query: str = Field(min_length=1, max_length=1500)
    limit: int = Field(default=5, ge=1, le=10)
    offset: int = Field(default=0, ge=0, le=9)
    refresh: bool = False


class FetchSourceAction(StrictModel):
    kind: Literal["fetch_source"]
    url: str = Field(min_length=1, max_length=2000)
    refresh: bool = False


class ReadSourceAction(StrictModel):
    kind: Literal["read_source"]
    source: str
    start_line: int = Field(default=1, ge=1)
    max_lines: int = Field(default=60, ge=1, le=150)
    max_characters: int = Field(default=5000, ge=500, le=6000)
    offset: int = Field(default=0, ge=0)


class FindSourceAction(StrictModel):
    kind: Literal["find_source"]
    source: str
    query: str = Field(min_length=1, max_length=500)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=10, ge=1, le=20)


class SearchRepositoriesAction(StrictModel):
    kind: Literal["search_repositories"]
    query: str = Field(min_length=1, max_length=1000)
    page: int = Field(default=1, ge=1, le=100)
    limit: int = Field(default=5, ge=1, le=20)
    refresh: bool = False


class ReadRepositoryAction(StrictModel):
    kind: Literal["read_repository"]
    repository: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    ref: str = Field(default="HEAD", min_length=1, max_length=100)
    path: str = Field(default="", max_length=500)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=40, ge=1, le=100)


class QueryExperimentsAction(StrictModel):
    kind: Literal["query_experiments"]
    target: Literal["trials", "candidates", "panels", "missing", "checks", "sources"] = "trials"
    candidate: str | None = None
    case: str | None = None
    steps: int | None = Field(default=None, gt=0)
    status: Literal["complete", "running", "failed"] | None = None
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=10, ge=1, le=30)


class CompareAction(StrictModel):
    kind: Literal["compare"]
    candidate: str
    steps: int = Field(gt=0)
    cases: list[str] | None = Field(default=None, min_length=1)
    train_seeds: list[int] | None = Field(default=None, min_length=1)
    eval_episodes: int | None = Field(default=None, ge=1)


class CheckCandidateAction(StrictModel):
    kind: Literal["check_candidate"]
    candidate: str
    case: str
    train_seed: int
    steps: int = Field(default=2, ge=1, le=100)


class RecallAction(StrictModel):
    kind: Literal["recall"]
    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=5, ge=1, le=20)


class RememberAction(StrictModel):
    kind: Literal["remember"]
    text: str = Field(min_length=1, max_length=3000)
    evidence_ids: list[str] = Field(min_length=1, max_length=20)
    category: Literal["hypothesis", "procedure", "literature", "decision"] = "hypothesis"
    supersedes: str | None = None


class FinishAction(StrictModel):
    kind: Literal["finish"]
    candidate: str
    conclusion: str = Field(min_length=1, max_length=5000)


Action = Annotated[
    PlanAction | ReviewProposalAction | ProposeAction | EditAlgorithmAction | RunAction | ContinueTrialAction | BranchStatusAction | ReadTrialLogAction | InspectAction | RecallAction | RememberAction | FinishAction |
    SearchLiteratureAction | SearchWebAction | FetchSourceAction | ReadSourceAction | FindSourceAction |
    SearchRepositoriesAction | ReadRepositoryAction | QueryExperimentsAction | CompareAction | CheckCandidateAction,
    Field(discriminator="kind"),
]


class Decision(StrictModel):
    """A short justification is an audit artifact, not a request for hidden reasoning."""
    summary: str = Field(min_length=1, max_length=2000)
    action: Action


class Feedback(StrictModel):
    status: Literal["ok", "invalid_action", "numerical_failure", "infrastructure_failure", "budget_stop"]
    summary: str
    evidence_ids: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class TrialResult(StrictModel):
    purpose: Literal["experiment", "check"] = "experiment"
    trial_id: str
    candidate: str
    case: str
    train_seed: int
    steps: int
    schedule_steps: int | None = None
    series_sha256: str = ""
    source_sha256: str = ""
    checkpoint_sha256: str = ""
    continued_from: str | None = None
    training_reused_from: str | None = None
    resumed_at: int = Field(default=0, ge=0)
    incremental_train_steps: int = Field(default=0, ge=0)
    eval_seeds: list[int] = Field(default_factory=list)
    returns: list[float] = Field(min_length=1)
    score: float
    metric: str
    artifact_sha256: str
    request_sha256: str
    train_seconds: float = Field(ge=0)
    evaluate_seconds: float = Field(ge=0)
    diagnostics: dict[str, float]
    evaluation_kind: str = Field(min_length=1, max_length=100)
    optimizer_steps: int = Field(default=0, ge=0)
    value_learning_audit: dict[str, Any] = Field(default_factory=dict)
    inference: dict[str, Any] = Field(default_factory=dict)
