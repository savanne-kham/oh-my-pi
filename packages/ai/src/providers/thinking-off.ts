import type { Api, Model } from "../types";
import { isAdaptiveOnlyThinking } from "./anthropic";

function usesApi<A extends Api>(model: Model, ...apis: A[]): model is Model<A> {
	return apis.some(api => model.api === api);
}

/**
 * True when turning thinking off for `model` really stops the model from
 * reasoning on the wire. Transports without such a switch run their lowest
 * effort instead, so callers that display or choose `off` must not offer it
 * there: the lowest effort is the honest label.
 *
 * - Anthropic: the model's declared `disabledThinking` when it has one (only
 *   `disabled` stops reasoning), else the legacy rule: models that accept
 *   `thinking: { type: "disabled" }`. Adaptive-only models pin the lowest
 *   effort, and `between_tools` still reasons between tool calls.
 * - Chat Completions: every disable dialect except `omit` and `lowest-effort`.
 * - Responses: the `none` tier and OpenRouter's `enabled: false`.
 * - Ollama: `think: false`.
 *
 * Models whose metadata requires an explicit effort never turn off.
 */
export function hasNativeThinkingOff(model: Model): boolean {
	if (!model.reasoning || model.thinking?.requiresEffort === true) return false;
	if (usesApi(model, "anthropic-messages")) {
		const { disabledThinking } = model.compat;
		if (disabledThinking !== undefined) return disabledThinking === "disabled";
		return !model.compat.supportsBetweenToolsThinking && !isAdaptiveOnlyThinking(model);
	}
	if (usesApi(model, "openai-completions")) {
		const mode = model.compat.reasoningDisableMode;
		return mode !== "omit" && mode !== "lowest-effort";
	}
	if (usesApi(model, "openai-responses", "azure-openai-responses", "openai-codex-responses")) {
		const mode = model.compat.reasoningDisableMode;
		return mode === "none-effort" || mode === "openrouter-enabled-false";
	}
	return model.api === "ollama-chat";
}
