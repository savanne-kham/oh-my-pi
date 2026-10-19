/**
 * Out-of-band vision routing for attached images. Text-only active models use
 * it automatically; `images.attachmentRouting: delegate` also keeps images out
 * of multimodal chat context. Each image is saved under the session `local://`
 * root, described through the vision route, and represented as text:
 *
 *     <image path="local://image-<hash>.png">
 *     <description>
 *     </image>
 */
import * as path from "node:path";
import {
	type AgentTelemetry,
	type AgentTelemetryConfig,
	instrumentedCompleteSimple,
	resolveTelemetry,
} from "@oh-my-pi/pi-agent-core";
import type { Api, completeSimple, ImageContent, Model, TextContent } from "@oh-my-pi/pi-ai";
import { sendsImageInputOnWire } from "@oh-my-pi/pi-ai/providers/vision-guard";
import { concreteThinkingLevel, resolveThinkingLevelForModel, toReasoningEffort } from "@oh-my-pi/pi-tui/thinking";
import { logger, prompt, toError } from "@oh-my-pi/pi-utils";
import { extractTextContent } from "../commit/utils";
import type { ModelRegistry } from "../config/model-registry";
import {
	expandRoleAlias,
	extractExplicitThinkingSelector,
	formatModelString,
	getModelMatchPreferences,
	resolveModelFromString,
	resolveModelOverride,
} from "../config/model-resolver";
import type { Settings } from "../config/settings";
import { type LocalProtocolOptions, resolveLocalRoot } from "../internal-urls";
import { cfgImagesBlockImages } from "../modes/settings";
import {
	findRetryFallbackCandidates,
	getRetryFallbackChains,
	type RetryFallbackResolutionContext,
	resolveRetryFallbackChainKey,
} from "../session/retry-fallback-chains";
import {
	cfgImagesAttachmentRouting,
	cfgImagesDescribeForTextModels,
	cfgRetryEnabled,
	cfgRetryModelFallback,
} from "../session/settings";
import describeUserPrompt from "../prompts/tools/image-attachment-describe.md" with { type: "text" };
import describeSystemPrompt from "../prompts/tools/image-attachment-describe-system.md" with { type: "text" };

/** Telemetry tag for the oneshot vision-description calls. */
const ONESHOT_KIND = "image_attachment_describe";

const NO_VISION_MODEL_NOTE =
	"[No vision-capable model is configured, so this image could not be described automatically. " +
	"The image was saved; configure a vision model role (modelRoles.vision) and read the saved image with ?q=<question> to analyze it.]";

const DESCRIPTION_UNAVAILABLE_NOTE =
	"[Image description unavailable: the vision model returned no usable text. The image was saved for further analysis.]";

/** Registry surface needed to resolve vision candidates and authorize requests. */
export type VisionFallbackRegistry = Pick<
	ModelRegistry,
	"find" | "getAvailable" | "getApiKey" | "hasProvider" | "resolver"
>;

/** Whether attached images should be described out-of-band instead of sent to the active model. */
export function shouldDescribeImageAttachments(model: Model<Api> | undefined, settings: Settings): model is Model<Api> {
	if (!model || cfgImagesBlockImages.get(settings)) return false;
	if (cfgImagesAttachmentRouting.get(settings) === "delegate") return true;
	return cfgImagesDescribeForTextModels.get(settings) && !sendsImageInputOnWire(model);
}

export interface DescribeAttachedImagesDeps {
	/** Active chat model, appended after configured vision fallbacks when it accepts images. */
	activeModel: Model<Api>;
	modelRegistry: VisionFallbackRegistry;
	settings: Settings;
	/** Inputs for resolving the session-scoped `local://` root. */
	localProtocolOptions: LocalProtocolOptions;
	/** `provider/id` selector for the active chat model. */
	activeModelString?: string;
	telemetryConfig?: AgentTelemetryConfig;
	sessionId?: string;
	/** Test seam: overrides the underlying completeSimple call. */
	completeImpl?: typeof completeSimple;
}

/** Map an image MIME type to a file extension for the saved artifact. */
function extensionForMime(mimeType: string): string {
	const subtype = mimeType.split("/")[1]?.toLowerCase() ?? "";
	switch (subtype) {
		case "jpeg":
		case "jpg":
			return "jpg";
		case "png":
			return "png";
		case "gif":
			return "gif";
		case "webp":
			return "webp";
		default: {
			const sanitized = subtype.replace(/[^a-z0-9]/g, "");
			return sanitized || "png";
		}
	}
}

/** Content-addressed file name so re-pasting the same image reuses one artifact. */
function imageFileName(image: ImageContent): string {
	const hash = Bun.hash(image.data).toString(16);
	return `image-${hash}.${extensionForMime(image.mimeType)}`;
}

/** Persist an image under the local root; returns its `local://` URL. */
async function saveImage(image: ImageContent, localRoot: string): Promise<string> {
	const fileName = imageFileName(image);
	const filePath = path.join(localRoot, fileName);
	// Content-addressed: identical bytes overwrite themselves harmlessly. Bun.write creates parent dirs.
	await Bun.write(filePath, Buffer.from(image.data, "base64"));
	return `local://${fileName}`;
}

function formatImageBlock(localUrl: string, description: string): string {
	return `<image path="${localUrl}">\n${description}\n</image>`;
}

interface VisionAttempt {
	model: Model<Api>;
	pattern: string | undefined;
}

/**
 * Resolve the delegated vision route: `@vision`, its configured fallback chain,
 * then the active chat model. Duplicate concrete models are removed.
 */
function resolveVisionAttempts(deps: DescribeAttachedImagesDeps): VisionAttempt[] {
	const available = deps.modelRegistry.getAvailable();
	if (available.length === 0) return [];
	const preferences = getModelMatchPreferences(deps.settings);
	const resolvePattern = (pattern: string | undefined): Model<Api> | undefined => {
		if (!pattern) return undefined;
		const expanded = expandRoleAlias(pattern, deps.settings);
		const model = resolveModelFromString(expanded, available, preferences);
		return model && sendsImageInputOnWire(model) ? model : undefined;
	};
	const attempts: VisionAttempt[] = [];
	const seen = new Set<string>();
	const add = (model: Model<Api> | undefined, pattern: string | undefined): void => {
		if (!model || !sendsImageInputOnWire(model)) return;
		const key = `${model.provider}/${model.id}`;
		if (seen.has(key)) return;
		seen.add(key);
		attempts.push({ model, pattern });
	};

	const visionPattern = deps.settings.getModelRole("vision");
	const primary = resolvePattern("@vision");
	add(primary, "@vision");

	if (primary && visionPattern && cfgRetryEnabled.get(deps.settings) && cfgRetryModelFallback.get(deps.settings)) {
		const context: RetryFallbackResolutionContext = {
			chains: getRetryFallbackChains(deps.settings),
			getModelRole: role => deps.settings.getModelRole(role),
			modelLookup: deps.modelRegistry,
		};
		const chainKey = resolveRetryFallbackChainKey(context, visionPattern, primary, "vision");
		if (chainKey) {
			for (const selector of findRetryFallbackCandidates(context, chainKey, visionPattern, primary)) {
				const resolved =
					resolveModelOverride([selector.raw], deps.modelRegistry, deps.settings).model ??
					deps.modelRegistry.find(selector.provider, selector.id);
				add(resolved, selector.raw);
			}
		}
	}

	// Preserve the old no-role fallback without letting it displace the active
	// model from the final slot of a configured vision route.
	if (!primary) {
		add(resolvePattern("@default"), "@default");
		add(
			available.find(candidate => sendsImageInputOnWire(candidate)),
			undefined,
		);
	}
	add(
		sendsImageInputOnWire(deps.activeModel) ? deps.activeModel : resolvePattern(deps.activeModelString),
		deps.activeModelString ?? formatModelString(deps.activeModel),
	);
	return attempts;
}

/** Run the delegated vision route; return the first usable description. */
async function describeImage(
	image: ImageContent,
	attempts: readonly VisionAttempt[],
	deps: DescribeAttachedImagesDeps,
	telemetry: AgentTelemetry | undefined,
	signal: AbortSignal | undefined,
): Promise<string | null> {
	for (const attempt of attempts) {
		const label = `${attempt.model.provider}/${attempt.model.id}`;
		try {
			if (!(await deps.modelRegistry.getApiKey(attempt.model, deps.sessionId))) {
				logger.warn("image attachment description skipped model without credentials", { model: label });
				continue;
			}
			const available = deps.modelRegistry.getAvailable();
			const configuredThinking = concreteThinkingLevel(
				extractExplicitThinkingSelector(attempt.pattern, deps.settings, {
					isLiteralModelId: (provider, id) => {
						if (provider !== undefined) {
							return available.some(candidate => candidate.provider === provider && candidate.id === id);
						}
						const lowerId = id.toLowerCase();
						return available.some(candidate => candidate.id.toLowerCase() === lowerId);
					},
				}),
			);
			const response = await instrumentedCompleteSimple(
				attempt.model,
				{
					systemPrompt: [prompt.render(describeSystemPrompt)],
					messages: [
						{
							role: "user",
							content: [
								{ type: "image", data: image.data, mimeType: image.mimeType },
								{ type: "text", text: prompt.render(describeUserPrompt) },
							],
							timestamp: Date.now(),
						},
					],
				},
				{
					apiKey: deps.modelRegistry.resolver(attempt.model, deps.sessionId),
					signal,
					reasoning: toReasoningEffort(resolveThinkingLevelForModel(attempt.model, configuredThinking)),
				},
				{ telemetry, oneshotKind: ONESHOT_KIND, completeImpl: deps.completeImpl },
			);
			if (response.stopReason === "aborted" && signal?.aborted) signal.throwIfAborted();
			if (response.stopReason === "error" || response.stopReason === "aborted") {
				logger.warn("image attachment description did not complete", {
					stopReason: response.stopReason,
					model: label,
				});
				continue;
			}
			const text = extractTextContent(response).trim();
			if (text.length > 0) return text;
			logger.warn("image attachment description returned no usable text", { model: label });
		} catch (error) {
			if (signal?.aborted) signal.throwIfAborted();
			logger.warn("image attachment description failed", {
				error: toError(error).message,
				model: label,
			});
		}
	}
	return null;
}

/**
 * Save each attached image under `local://` and replace it with a descriptive
 * text block. The vision role is tried first, followed by its configured
 * fallback chain and finally the active chat model when it accepts images.
 */
export async function describeAttachedImages(
	images: readonly ImageContent[],
	deps: DescribeAttachedImagesDeps,
	signal?: AbortSignal,
): Promise<TextContent[]> {
	const localRoot = resolveLocalRoot(deps.localProtocolOptions);
	const attempts = resolveVisionAttempts(deps);
	const telemetry = resolveTelemetry(deps.telemetryConfig, deps.sessionId);

	return Promise.all(
		images.map(async (image): Promise<TextContent> => {
			const localUrl = await saveImage(image, localRoot);
			const description =
				attempts.length > 0
					? ((await describeImage(image, attempts, deps, telemetry, signal)) ?? DESCRIPTION_UNAVAILABLE_NOTE)
					: NO_VISION_MODEL_NOTE;
			return { type: "text", text: formatImageBlock(localUrl, description) };
		}),
	);
}
