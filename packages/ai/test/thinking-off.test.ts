import { describe, expect, it } from "bun:test";
import { hasNativeThinkingOff } from "@oh-my-pi/pi-ai/providers/thinking-off";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";

describe("hasNativeThinkingOff", () => {
	it("is true where the transport really stops reasoning", () => {
		expect(hasNativeThinkingOff(getBundledModel("anthropic", "claude-haiku-5-5"))).toBe(true);
		expect(hasNativeThinkingOff(getBundledModel("openai-codex", "gpt-5.6-terra"))).toBe(true);
		expect(hasNativeThinkingOff(getBundledModel("opencode-go", "deepseek-v4-pro"))).toBe(true);
		expect(hasNativeThinkingOff(getBundledModel("opencode-go", "qwen3.8-flash"))).toBe(true);
	});

	it("is false where off would run the lowest effort or still reason", () => {
		// Adaptive-only Claude pins the lowest effort; Sonnet 5.5 only skips thinking before the answer.
		expect(hasNativeThinkingOff(getBundledModel("anthropic", "claude-opus-4-6"))).toBe(false);
		expect(hasNativeThinkingOff(getBundledModel("anthropic", "claude-opus-5-5"))).toBe(false);
		expect(hasNativeThinkingOff(getBundledModel("anthropic", "claude-sonnet-5-5"))).toBe(false);
		// Effort-requiring models have no off at all.
		expect(hasNativeThinkingOff(getBundledModel("opencode-go", "kimi-k3"))).toBe(false);
		expect(hasNativeThinkingOff(getBundledModel("google-antigravity", "gemini-3.8-flash"))).toBe(false);
	});
});
