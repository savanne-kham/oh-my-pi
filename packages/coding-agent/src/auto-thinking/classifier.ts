/**
 * Per-prompt difficulty classifier for the `auto` thinking level.
 *
 * Asks a {@link ChoiceQuestion} when multiple supported online tiers exist,
 * short-circuiting without a judge request when zero or one tier is available.
 * For task-spawned turns, it considers the delegator's `solutionSpace` alone.
 * It maps the selected level to a concrete {@link Effort} and clamps it into
 * the active model's supported range (never below {@link Effort.Low}), or to
 * {@link ThinkingLevel.Off} when `providers.autoThinkingAllowOff` is on and the
 * judge finds no reasoning to do.
 * The judge comes from the live `judge` role chain. A local on-device candidate
 * gets the coarser
 * `trivial|moderate|hard` question (3-class is more reliable
 * than 4-way ordinal on sub-2B models), mapped to `low|high|xhigh`; it never
 * selects off.
 *
 * Throws on any failure (no judge, no key, unparseable output, abort/timeout);
 * the caller falls back to a concrete level and continues the turn.
 */
import { type ResolvedThinkingLevel, ThinkingLevel, type AgentTelemetryConfig } from "@oh-my-pi/pi-agent-core";
import { type ChoiceQuestion, Effort, judgeQuestions, type Model, renderQuestion } from "@oh-my-pi/pi-ai";
import { hasNativeThinkingOff } from "@oh-my-pi/pi-ai/providers/thinking-off";
import { getSupportedEfforts } from "@oh-my-pi/pi-catalog/model-thinking";
import type { ModelRegistry } from "../config/model-registry";
import type { Settings } from "../config/settings";
import { type JudgmentUsage, resolveJudge, sharedJudgmentCache } from "../judgment";
import questionFile from "../prompts/judge/auto-thinking.json" with { type: "json" };
import { clampAutoThinkingEffort } from "@oh-my-pi/pi-tui/thinking";
import { preprocessTinyMessage } from "../tiny/message-preproc";

import { cfgProvidersAutoThinkingAllowOff, cfgProvidersAutoThinkingMaxEffort } from "../session/settings";

type EffortLevel = "low" | "medium" | "high" | "xhigh" | "max";
type Level = "off" | EffortLevel;
type Bucket = "trivial" | "moderate" | "hard";

const LEVEL_EFFORT: Record<EffortLevel, Effort> = {
	low: Effort.Low,
	medium: Effort.Medium,
	high: Effort.High,
	xhigh: Effort.XHigh,
	max: Effort.Max,
};

const BUCKET_EFFORT: Record<Bucket, Effort> = {
	trivial: Effort.Low,
	moderate: Effort.High,
	hard: Effort.XHigh,
};

/**
 * The on-device bucket question, the effort ladder (levels by how open-ended
 * the problem is, `off` first and `max` last), and the solution-space
 * instructions, which reuse the ladder's levels.
 */
const QUESTIONS = judgeQuestions(questionFile);

/**
 * Ladder variants by which optional ends they offer: `max` only when the
 * target model exposes that tier and the ceiling allows it, `off` only when the
 * user enables `providers.autoThinkingAllowOff` and the model can disable reasoning.
 */
type LadderVariant = "core" | "withMax" | "withOff" | "withMaxAndOff";

/** Questions for one classification input kind: on-device bucket, plus each ladder variant. */
interface QuestionSet {
	bucket: ChoiceQuestion<Bucket>;
	ladder: Record<LadderVariant, ChoiceQuestion>;
}

/**
 * Build the question set for one input kind. Only the instructions differ
 * between kinds; every kind shares the ladder's levels and the bucket criteria.
 * A variant that does not offer an end drops both its criterion and its
 * instruction clause.
 */
function buildQuestionSet(instructions: string, bucketInstructions: string): QuestionSet {
	const ladder: ChoiceQuestion<Level> = { ...QUESTIONS.level, instructions };
	const { off, max, ...core } = ladder.criteria;
	const variant = (withMax: boolean, withOff: boolean): ChoiceQuestion => ({
		...renderQuestion(ladder, { withMax, withOff }),
		criteria: { ...(withOff && { off }), ...core, ...(withMax && { max }) },
	});
	return {
		bucket: { ...QUESTIONS.bucket, instructions: bucketInstructions },
		ladder: {
			core: variant(false, false),
			withMax: variant(true, false),
			withOff: variant(false, true),
			withMaxAndOff: variant(true, true),
		},
	};
}

function isEffortLevel(label: string): label is EffortLevel {
	return Object.hasOwn(LEVEL_EFFORT, label);
}

/** The ladder question narrowed to the levels the target model supports. */
interface SupportedLevelQuestion {
	question: ChoiceQuestion;
	/** Supported effort levels, lowest first. `off` is never listed here. */
	efforts: EffortLevel[];
	/** Whether the question still offers `off`; only alongside at least one effort. */
	offersOff: boolean;
}

/** Keep the judge's choices aligned with the effort surface the target can use. */
function filterSupportedLevels(question: ChoiceQuestion, model: Model): SupportedLevelQuestion {
	const supported = getSupportedEfforts(model);
	const criteria: Record<string, string | null> = {};
	const efforts: EffortLevel[] = [];
	for (const [label, rubric] of Object.entries(question.criteria)) {
		if (!isEffortLevel(label) || !supported.includes(LEVEL_EFFORT[label])) continue;
		criteria[label] = rubric;
		efforts.push(label);
	}
	// `off` is a disable state beside the ladder, never the ladder itself: with no
	// effort to weigh it against it would turn every turn off unasked.
	const offersOff = Object.hasOwn(question.criteria, "off") && efforts.length > 0;
	return {
		question: { ...question, criteria: offersOff ? { off: question.criteria.off, ...criteria } : criteria },
		efforts,
		offersOff,
	};
}

const REQUEST_QUESTIONS = buildQuestionSet(QUESTIONS.level.instructions, QUESTIONS.bucket.instructions);
const SOLUTION_SPACE_QUESTIONS = buildQuestionSet(
	QUESTIONS.solutionSpace.instructions,
	renderQuestion(QUESTIONS.solutionSpace, {}).instructions,
);

/** The turn to classify. */
export interface DifficultyInput {
	/** The prompt text the agent is about to act on. */
	request: string;
	/**
	 * Delegator's description of how open-ended the subtask is (task `solutionSpace` field).
	 * When non-blank it replaces `request` as the sole classification input.
	 */
	solutionSpace?: string;
}

export interface ClassifyDifficultyDeps {
	settings: Settings;
	registry: ModelRegistry;
	model: Model;
	sessionId?: string;
	signal?: AbortSignal;
	metadataResolver?: (provider: string) => Record<string, unknown> | undefined;
	onUsage?: (usage: JudgmentUsage) => void;
	telemetry?: AgentTelemetryConfig;
}

/**
 * Highest effort this turn's classification may resolve to: the configured
 * ceiling, further limited by what the target model actually exposes. The
 * default keeps `auto` one tier below the top, so only an explicit
 * `ultrathink` reaches {@link Effort.Max}.
 */
function autoEffortCeiling(deps: ClassifyDifficultyDeps): Effort {
	if (cfgProvidersAutoThinkingMaxEffort.get(deps.settings) !== Effort.Max) return Effort.XHigh;
	return getSupportedEfforts(deps.model).includes(Effort.Max) ? Effort.Max : Effort.XHigh;
}

/**
 * Classify `input` and return a concrete effort clamped to `deps.model`,
 * {@link ThinkingLevel.Off} when `providers.autoThinkingAllowOff` is on and the judge
 * chose it, or `undefined` when the model has no controllable effort surface
 * or no supported effort under the active ceiling (auto has nothing to pick —
 * the caller falls back to its provisional auto level).
 * @throws when the backend cannot produce a usable classification.
 */
export async function classifyDifficulty(
	input: DifficultyInput,
	deps: ClassifyDifficultyDeps,
): Promise<ResolvedThinkingLevel | undefined> {
	const judge = resolveJudge({
		settings: deps.settings,
		registry: deps.registry,
		sessionModel: deps.model,
		sessionId: deps.sessionId,
		metadataResolver: deps.metadataResolver,
		purpose: "auto-thinking",
		onUsage: deps.onUsage,
		telemetry: deps.telemetry,
		cache: sharedJudgmentCache(),
	});
	const solutionSpace = input.solutionSpace?.trim();
	const state: Record<string, string> = solutionSpace
		? { solution_space: preprocessTinyMessage(solutionSpace) }
		: { request: preprocessTinyMessage(input.request) };
	const questions = solutionSpace ? SOLUTION_SPACE_QUESTIONS : REQUEST_QUESTIONS;
	const options = { signal: deps.signal };
	const ceiling = autoEffortCeiling(deps);
	// `auto` may turn thinking off only when the user opts in and the transport
	// really disables reasoning. Elsewhere `off` would run the lowest effort under
	// a misleading label, so the judge picks that effort by its real name.
	const withOff = cfgProvidersAutoThinkingAllowOff.get(deps.settings) && hasNativeThinkingOff(deps.model);
	const withMax = ceiling === Effort.Max;
	const variant: LadderVariant = withMax ? (withOff ? "withMaxAndOff" : "withMax") : withOff ? "withOff" : "core";
	const level = filterSupportedLevels(questions.ladder[variant], deps.model);
	const [onlyEffort, ...otherEfforts] = level.efforts;
	// One eligible tier below Max is the answer every candidate reaches (the local
	// bucket clamps into the same pool), so it needs no judge and no judge credential.
	// Offering `off` beside it is a real choice and still goes to the judge.
	if (onlyEffort !== undefined && otherEfforts.length === 0 && onlyEffort !== "max" && !level.offersOff) {
		return LEVEL_EFFORT[onlyEffort];
	}
	// With no online tier, every candidate lands in the same place: the local
	// bucket's XHigh-capped pool, which is empty (`["max"]`) or holds only a
	// sub-Low tier (`["minimal"]`).
	if (onlyEffort === undefined) return clampAutoThinkingEffort(deps.model, Effort.Low, Effort.XHigh);
	const classified = await judge.withCandidate(async (candidate, kind) => {
		// The 3-bucket local question cannot select `max`, so its ceiling stays at
		// XHigh whatever the setting says — otherwise a sparse ladder would snap its
		// `hard` bucket up to a tier it never chose. It cannot select `off` either.
		if (kind === "local") {
			const { answers } = await candidate.judge({ state, questions: { bucket: questions.bucket } }, options);
			return { level: BUCKET_EFFORT[answers.bucket.choice], ceiling: Effort.XHigh };
		}
		if (otherEfforts.length === 0 && !level.offersOff) return { level: LEVEL_EFFORT[onlyEffort], ceiling };
		const { answers } = await candidate.judge({ state, questions: { level: level.question } }, options);
		const choice = answers.level.choice;
		if (choice === "off" && level.offersOff) return { level: ThinkingLevel.Off, ceiling };
		const chosen = level.efforts.find(label => label === choice);
		if (chosen === undefined) {
			throw new Error(`auto-thinking: judge chose unsupported level "${choice}"`);
		}
		return { level: LEVEL_EFFORT[chosen], ceiling };
	}, options);
	if (classified.level === ThinkingLevel.Off) return ThinkingLevel.Off;
	// Local bucket choices are unfiltered and may fall below every supported tier.
	// Passing the ceiling prevents snapping upward into an excluded tier.
	return clampAutoThinkingEffort(deps.model, classified.level, classified.ceiling);
}
