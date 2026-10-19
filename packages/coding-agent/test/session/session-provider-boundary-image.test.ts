import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import type { AgentMessage } from "@oh-my-pi/pi-agent-core";
import type { Api, ImageContent, Message, Model } from "@oh-my-pi/pi-ai";
import type { ModelSpec } from "@oh-my-pi/pi-ai/types";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import {
	SessionProviderBoundary,
	type SessionProviderBoundaryHost,
} from "@oh-my-pi/pi-coding-agent/session/session-provider-boundary";
import { removeWithRetries } from "@oh-my-pi/pi-utils";

const TINY_PNG_BASE64 =
	"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==";

function makeProxyModel(id: string, compat?: ModelSpec["compat"]): Model<Api> {
	return buildModel({
		id,
		name: id,
		api: "openai-completions",
		provider: "myproxy",
		baseUrl: "https://proxy.example.com/v1",
		reasoning: false,
		input: ["text", "image"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128_000,
		maxTokens: 8_192,
		compat,
	} as ModelSpec);
}

function makeHost(
	active: Model<Api>,
	artifactsDir: string,
	settings = Settings.isolated(),
): SessionProviderBoundaryHost {
	return {
		agent: { telemetry: undefined },
		sessionManager: {},
		settings,
		modelRegistry: {
			getAvailable: () => [],
		},
		model: () => active,
		sessionId: () => "test-session",
		localProtocolOptions: () => ({ getArtifactsDir: () => artifactsDir, getSessionId: () => "test-session" }),
		transformContext: (messages: AgentMessage[]) => messages,
		convertToLlm: (messages: AgentMessage[]) => messages as unknown as Message[],
		obfuscator: () => undefined,
	} as unknown as SessionProviderBoundaryHost;
}

describe("buildImageDescriptionNotice wire truth (#9697)", () => {
	let testDir: string;

	beforeEach(async () => {
		testDir = await fs.mkdtemp(path.join(os.tmpdir(), "boundary-wire-"));
	});

	afterEach(async () => {
		await removeWithRetries(testDir);
	});

	it("falls back for a wire-stripped model that declares image input, not for an opted-out one", async () => {
		const image: ImageContent = { type: "image", data: TINY_PNG_BASE64, mimeType: "image/png" };
		const stripped = makeProxyModel("deepseek-v4-flash", { stripImageInput: true });
		const notice = await new SessionProviderBoundary(makeHost(stripped, testDir)).buildImageDescriptionNotice([
			image,
		]);
		expect(notice).toBeDefined();
		expect(JSON.stringify(notice)).toContain("No vision-capable model");

		const optedOut = makeProxyModel("deepseek-v4-flash", { stripImageInput: false });
		const silent = await new SessionProviderBoundary(makeHost(optedOut, testDir)).buildImageDescriptionNotice([
			image,
		]);
		expect(silent).toBeUndefined();
	});

	it("delegates attachments for a vision-capable model when explicitly configured", async () => {
		const image: ImageContent = { type: "image", data: TINY_PNG_BASE64, mimeType: "image/png" };
		const active = makeProxyModel("vision-chat", { stripImageInput: false });
		const settings = Settings.isolated({
			"images.attachmentRouting": "delegate",
			"images.describeForTextModels": false,
		});
		const notice = await new SessionProviderBoundary(makeHost(active, testDir, settings)).buildImageDescriptionNotice(
			[image],
		);

		expect(notice).toBeDefined();
		expect(JSON.stringify(notice)).toContain("No vision-capable model");
	});

	it("keeps attachment bytes in session state but strips them from delegated provider context", async () => {
		const image: ImageContent = { type: "image", data: TINY_PNG_BASE64, mimeType: "image/png" };
		const active = makeProxyModel("vision-chat", { stripImageInput: false });
		const settings = Settings.isolated({ "images.attachmentRouting": "delegate" });
		const boundary = new SessionProviderBoundary(makeHost(active, testDir, settings));
		const messages = [
			{
				role: "user" as const,
				content: [{ type: "text" as const, text: "Inspect this." }, image],
				timestamp: 1,
			},
		];

		const converted = await boundary.convertMessagesToLlm(messages);

		expect(messages[0]!.content).toContain(image);
		expect(JSON.stringify(converted)).not.toContain(TINY_PNG_BASE64);
		expect(JSON.stringify(converted)).toContain("Inspect this.");
	});
});
