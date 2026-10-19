import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import type { Api, AssistantMessage, completeSimple, Model } from "@oh-my-pi/pi-ai";
import type { ModelSpec } from "@oh-my-pi/pi-ai/types";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import {
	type DescribeAttachedImagesDeps,
	describeAttachedImages,
} from "@oh-my-pi/pi-coding-agent/utils/image-vision-fallback";
import { removeWithRetries } from "@oh-my-pi/pi-utils";

// 1x1 transparent PNG.
const TINY_PNG_BASE64 =
	"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==";

const visionModel: Model<"openai-responses"> = buildModel({
	id: "gpt-4o",
	name: "GPT-4o",
	api: "openai-responses",
	provider: "openai",
	baseUrl: "https://api.openai.com/v1",
	reasoning: false,
	input: ["text", "image"],
	cost: { input: 5, output: 15, cacheRead: 0.5, cacheWrite: 5 },
	contextWindow: 128000,
	maxTokens: 4096,
});

const textModel: Model<"openai-responses"> = { ...visionModel, id: "gpt-4.1-mini", input: ["text"] };

function makeCompleteStub(text: string): { calls: unknown[][]; fn: typeof completeSimple } {
	const calls: unknown[][] = [];
	const fn = (async (...args: unknown[]) => {
		calls.push(args);
		return {
			role: "assistant",
			api: visionModel.api,
			provider: visionModel.provider,
			model: visionModel.id,
			usage: {
				input: 1,
				output: 1,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 2,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "stop",
			timestamp: Date.now(),
			content: [{ type: "text", text }],
		} satisfies AssistantMessage;
	}) as typeof completeSimple;
	return { calls, fn };
}

function makeRoutingStub(successId: string, text: string): { modelIds: string[]; fn: typeof completeSimple } {
	const modelIds: string[] = [];
	const fn = (async (...args: unknown[]) => {
		const selected = args[0] as Model<Api>;
		modelIds.push(selected.id);
		const usage = {
			input: 1,
			output: 1,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 2,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		};
		if (selected.id !== successId) {
			return {
				role: "assistant",
				api: selected.api,
				provider: selected.provider,
				model: selected.id,
				usage,
				stopReason: "error",
				errorMessage: "rate limit",
				timestamp: Date.now(),
				content: [],
			} satisfies AssistantMessage;
		}
		return {
			role: "assistant",
			api: selected.api,
			provider: selected.provider,
			model: selected.id,
			usage,
			stopReason: "stop",
			timestamp: Date.now(),
			content: [{ type: "text", text }],
		} satisfies AssistantMessage;
	}) as typeof completeSimple;
	return { modelIds, fn };
}

function makeDeps(
	artifactsDir: string,
	available: Model<Api>[],
	completeImpl: typeof completeSimple,
	apiKey: string | undefined = "test-key",
	activeModel: Model<Api> = textModel,
	settings = Settings.isolated(),
): DescribeAttachedImagesDeps {
	return {
		activeModel,
		modelRegistry: {
			getAvailable: () => available,
			getApiKey: async () => apiKey,
			find: (provider: string, id: string) =>
				available.find(candidate => candidate.provider === provider && candidate.id === id),
			hasProvider: (provider: string) => available.some(candidate => candidate.provider === provider),
			resolver: () => async () => apiKey,
		} as unknown as DescribeAttachedImagesDeps["modelRegistry"],
		settings,
		localProtocolOptions: { getArtifactsDir: () => artifactsDir, getSessionId: () => "test-session" },
		activeModelString: `${activeModel.provider}/${activeModel.id}`,
		completeImpl,
	};
}

describe("describeAttachedImages", () => {
	let testDir: string;

	beforeEach(async () => {
		testDir = await fs.mkdtemp(path.join(os.tmpdir(), "vision-fallback-"));
	});

	afterEach(async () => {
		await removeWithRetries(testDir);
	});

	it("saves the image under local:// and injects a vision description block", async () => {
		const stub = makeCompleteStub("A man holding a red balloon.");
		const blocks = await describeAttachedImages(
			[{ type: "image", data: TINY_PNG_BASE64, mimeType: "image/png" }],
			makeDeps(testDir, [textModel, visionModel], stub.fn),
		);

		expect(blocks).toHaveLength(1);
		const text = blocks[0]!.text;
		// Block wraps the local:// path and the description.
		const match = text.match(/^<image path="(local:\/\/[^"]+\.png)">\n([\s\S]*)\n<\/image>$/);
		expect(match).not.toBeNull();
		expect(match![2]).toBe("A man holding a red balloon.");

		// The vision model was actually consulted.
		expect(stub.calls).toHaveLength(1);

		// The image was persisted under <artifactsDir>/local and round-trips.
		const fileName = match![1].slice("local://".length);
		const saved = await fs.readFile(path.join(testDir, "local", fileName));
		expect(saved.toString("base64")).toBe(TINY_PNG_BASE64);
	});

	it("walks the vision fallback chain after a provider error", async () => {
		const fallbackModel: Model<Api> = {
			...visionModel,
			provider: "fallback",
			id: "vision-fallback",
			name: "Vision Fallback",
		};
		const activeModel: Model<Api> = {
			...visionModel,
			provider: "chat",
			id: "active-chat",
			name: "Active Chat",
		};
		const settings = Settings.isolated({
			"retry.fallbackChains": { vision: [`${fallbackModel.provider}/${fallbackModel.id}:low`] },
		});
		settings.setModelRole("vision", `${visionModel.provider}/${visionModel.id}:medium`);
		const stub = makeRoutingStub(fallbackModel.id, "Fallback description.");

		const blocks = await describeAttachedImages(
			[{ type: "image", data: TINY_PNG_BASE64, mimeType: "image/png" }],
			makeDeps(testDir, [visionModel, fallbackModel, activeModel], stub.fn, "test-key", activeModel, settings),
		);

		expect(stub.modelIds).toEqual([visionModel.id, fallbackModel.id]);
		expect(blocks[0]!.text).toContain("Fallback description.");
	});

	it("uses the active chat model only after every configured vision model fails", async () => {
		const fallbackModel: Model<Api> = {
			...visionModel,
			provider: "fallback",
			id: "vision-fallback",
			name: "Vision Fallback",
		};
		const activeModel: Model<Api> = {
			...visionModel,
			provider: "chat",
			id: "active-chat",
			name: "Active Chat",
		};
		const settings = Settings.isolated({
			"retry.fallbackChains": { vision: [`${fallbackModel.provider}/${fallbackModel.id}`] },
		});
		settings.setModelRole("vision", `${visionModel.provider}/${visionModel.id}`);
		const stub = makeRoutingStub(activeModel.id, "Active model description.");

		const blocks = await describeAttachedImages(
			[{ type: "image", data: TINY_PNG_BASE64, mimeType: "image/png" }],
			makeDeps(testDir, [visionModel, fallbackModel, activeModel], stub.fn, "test-key", activeModel, settings),
		);

		expect(stub.modelIds).toEqual([visionModel.id, fallbackModel.id, activeModel.id]);
		expect(blocks[0]!.text).toContain("Active model description.");
	});

	it("saves the image but emits a no-vision note when no vision model is available", async () => {
		const stub = makeCompleteStub("should not be used");
		const blocks = await describeAttachedImages(
			[{ type: "image", data: TINY_PNG_BASE64, mimeType: "image/png" }],
			makeDeps(testDir, [textModel], stub.fn),
		);

		expect(blocks).toHaveLength(1);
		// No vision-capable model -> no model call, but a note + saved artifact.
		expect(stub.calls).toHaveLength(0);
		expect(blocks[0]!.text).toContain("No vision-capable model");

		const match = blocks[0]!.text.match(/path="local:\/\/([^"]+)"/);
		expect(match).not.toBeNull();
		const saved = await fs.readFile(path.join(testDir, "local", match![1]));
		expect(saved.toString("base64")).toBe(TINY_PNG_BASE64);
	});

	it("falls back to a note when the vision model returns no text", async () => {
		const stub = makeCompleteStub("   ");
		const blocks = await describeAttachedImages(
			[{ type: "image", data: TINY_PNG_BASE64, mimeType: "image/png" }],
			makeDeps(testDir, [textModel, visionModel], stub.fn),
		);

		expect(stub.calls).toHaveLength(1);
		expect(blocks[0]!.text).toContain("Image description unavailable");
	});

	it("is content-addressed: identical images reuse one saved file path", async () => {
		const image = { type: "image", data: TINY_PNG_BASE64, mimeType: "image/png" } as const;
		const stub = makeCompleteStub("desc");
		const blocks = await describeAttachedImages([image, image], makeDeps(testDir, [textModel, visionModel], stub.fn));

		const paths = blocks.map(b => b.text.match(/path="(local:\/\/[^"]+)"/)![1]);
		expect(paths[0]).toBe(paths[1]);
	});
	it("treats a wire-stripped completions model as text-only when resolving the vision model", async () => {
		const stripped = buildModel({
			id: "deepseek-v4-flash",
			name: "deepseek-v4-flash",
			api: "openai-completions",
			provider: "myproxy",
			baseUrl: "https://proxy.example.com/v1",
			reasoning: false,
			input: ["text", "image"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 128_000,
			maxTokens: 8_192,
			compat: { stripImageInput: true },
		} as ModelSpec);
		const stub = makeCompleteStub("should not be used");
		const blocks = await describeAttachedImages(
			[{ type: "image", data: TINY_PNG_BASE64, mimeType: "image/png" }],
			makeDeps(testDir, [stripped], stub.fn),
		);
		expect(stub.calls).toHaveLength(0);
		expect(blocks).toHaveLength(1);
		expect(blocks[0]!.text).toContain("No vision-capable model");
	});
});
