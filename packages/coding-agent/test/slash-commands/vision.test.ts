import { describe, expect, it, vi } from "bun:test";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { executeAcpBuiltinSlashCommand } from "@oh-my-pi/pi-coding-agent/slash-commands/acp-builtins";
import { cfgImagesAttachmentRouting } from "@oh-my-pi/pi-coding-agent/session/settings";

function acpRuntime(initial: "delegate" | "native" = "native") {
	const settings = Settings.isolated({ "images.attachmentRouting": initial });
	const output = vi.fn();
	return { output, settings, runtime: { session: { settings }, output } };
}

describe("/vision slash command", () => {
	it("switches pasted images to delegated routing without persisting", async () => {
		const h = acpRuntime();

		expect(await Reflect.apply(executeAcpBuiltinSlashCommand, undefined, ["/vision delegate", h.runtime])).toEqual({
			consumed: true,
		});
		expect(cfgImagesAttachmentRouting.get(h.settings)).toBe("delegate");
		expect(h.settings.getGlobalSettings()).toEqual({});
		expect(h.output).toHaveBeenCalledWith(
			"Image attachment routing: delegate for this session. Pasted images go through @vision, its fallback chain, then the active chat model.",
		);
	});

	it("restores native routing for the current session", async () => {
		const h = acpRuntime("delegate");
		await Reflect.apply(executeAcpBuiltinSlashCommand, undefined, ["/vision native", h.runtime]);

		expect(cfgImagesAttachmentRouting.get(h.settings)).toBe("native");
		expect(h.output).toHaveBeenCalledWith(
			"Image attachment routing: native for this session. Vision-capable chat models receive pasted images directly; text-only models still delegate.",
		);
	});

	it("reports status and rejects unknown routes", async () => {
		const status = acpRuntime("delegate");
		await Reflect.apply(executeAcpBuiltinSlashCommand, undefined, ["/vision status", status.runtime]);
		expect(status.output).toHaveBeenCalledWith("Image attachment routing: delegate.");

		const invalid = acpRuntime();
		await Reflect.apply(executeAcpBuiltinSlashCommand, undefined, ["/vision direct", invalid.runtime]);
		expect(invalid.output).toHaveBeenCalledWith("Usage: /vision [delegate|native|status]");
	});
});
