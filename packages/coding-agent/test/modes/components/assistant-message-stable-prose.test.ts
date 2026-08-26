import { beforeAll, beforeEach, describe, expect, it } from "bun:test";
import type { AssistantMessage } from "@oh-my-pi/pi-ai";
import { resetSettingsForTest, Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { AssistantMessageComponent } from "@oh-my-pi/pi-coding-agent/modes/components/assistant-message";
import { TranscriptContainer } from "@oh-my-pi/pi-coding-agent/modes/components/transcript-container";
import { initTheme } from "@oh-my-pi/pi-coding-agent/modes/theme/theme";

function createAssistantMessage(markdown: string): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text: markdown }],
		api: "anthropic-messages",
		provider: "anthropic",
		model: "claude-sonnet-4-5",
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
		timestamp: Date.now(),
	};
}

function thinkingThenText(thinking: string, text: string): AssistantMessage {
	const message = createAssistantMessage(text);
	return { ...message, content: [{ type: "thinking", thinking }, ...message.content] };
}

/** Drain every currently offered mid-stream stable-row batch under pressure. */
function drainPressureBatches(transcript: TranscriptContainer, width: number): string[] {
	const batches: string[] = [];
	for (;;) {
		const batch = transcript.peekFinalizedBatch(width, 0);
		if (batch === undefined) return batches;
		batches.push(Bun.stripANSI(batch.rows.join("\n")));
		transcript.acknowledgeFinalizedBatch(batch.id);
	}
}

function occurrences(haystack: string, needle: string): number {
	let count = 0;
	for (let index = haystack.indexOf(needle); index >= 0; index = haystack.indexOf(needle, index + needle.length)) {
		count++;
	}
	return count;
}

beforeAll(async () => {
	await initTheme(false);
});

beforeEach(async () => {
	resetSettingsForTest();
	await Settings.init({ inMemory: true });
});

describe("AssistantMessageComponent stable prose publication", () => {
	it("retires settled prose paragraphs into history exactly once, in order, while streaming", () => {
		const component = new AssistantMessageComponent();
		const transcript = new TranscriptContainer();
		transcript.addChild(component);
		let tick = 0;
		const render = () => transcript.renderViewport(80, 20, { now: tick, tick: tick++ });

		const paragraphs = [
			"PROSE-1 opening paragraph with enough words to be a real row.",
			"PROSE-2 second paragraph continues the streamed answer.",
			"PROSE-3 third paragraph keeps the stream going.",
			"PROSE-4 fourth paragraph arrives later in the stream.",
		];
		component.updateContent(
			createAssistantMessage(`${paragraphs.slice(0, 3).join("\n\n")}\n\nTAIL-PARTIAL still streaming`),
			{ transient: true },
		);
		render();

		const early = drainPressureBatches(transcript, 80);
		expect(early.length).toBeGreaterThan(0);
		const earlyText = early.join("\n");
		expect(earlyText).toContain("PROSE-1");
		expect(earlyText).toContain("PROSE-3");
		expect(earlyText).not.toContain("TAIL-PARTIAL");

		// Emitted rows leave the mutable viewport; the live tail stays.
		const live = Bun.stripANSI(render().join("\n"));
		expect(live).not.toContain("PROSE-1");
		expect(live).toContain("TAIL-PARTIAL");

		// The stream grows: the next settled paragraph retires as a delta batch.
		component.updateContent(
			createAssistantMessage(`${paragraphs.join("\n\n")}\n\nTAIL-PARTIAL still streaming longer`),
			{ transient: true },
		);
		render();
		const later = drainPressureBatches(transcript, 80);
		const laterText = later.join("\n");
		expect(laterText).toContain("PROSE-4");
		expect(laterText).not.toContain("PROSE-3");

		// Finalization flushes only the un-emitted suffix.
		const finalMessage = createAssistantMessage(`${paragraphs.join("\n\n")}\n\nTAIL-PARTIAL still streaming longer`);
		component.updateContent(finalMessage);
		component.markTranscriptBlockFinalized();
		render();
		const flush = transcript.peekFlushBatch(80);
		expect(flush).toBeDefined();
		transcript.acknowledgeFinalizedBatch(flush!.id);
		const flushText = Bun.stripANSI(flush!.rows.join("\n"));
		expect(flushText).toContain("TAIL-PARTIAL");

		const everything = [earlyText, laterText, flushText].join("\n");
		for (const marker of ["PROSE-1", "PROSE-2", "PROSE-3", "PROSE-4", "TAIL-PARTIAL"]) {
			expect(occurrences(everything, marker)).toBe(1);
		}
		const order = ["PROSE-1", "PROSE-2", "PROSE-3", "PROSE-4", "TAIL-PARTIAL"].map(marker =>
			everything.indexOf(marker),
		);
		expect([...order].sort((a, b) => a - b)).toEqual(order);
	});

	it("publishes visible thinking before prose in transcript order without duplicates", () => {
		const component = new AssistantMessageComponent();
		const transcript = new TranscriptContainer();
		transcript.addChild(component);
		let tick = 0;
		const render = () => transcript.renderViewport(80, 20, { now: tick, tick: tick++ });

		component.updateContent(
			thinkingThenText("THINK-ALPHA reasoning paragraph.\n\nTHINK-BETA reasoning paragraph.", ""),
			{ transient: true },
		);
		render();
		component.updateContent(
			thinkingThenText(
				"THINK-ALPHA reasoning paragraph.\n\nTHINK-BETA reasoning paragraph.",
				"ANSWER-1 first answer paragraph.\n\nANSWER-2 partial tail",
			),
			{ transient: true },
		);
		render();

		const streamed = drainPressureBatches(transcript, 80).join("\n");
		expect(streamed).toContain("THINK-ALPHA");
		expect(streamed).toContain("ANSWER-1");
		expect(streamed).not.toContain("ANSWER-2");

		const final = thinkingThenText(
			"THINK-ALPHA reasoning paragraph.\n\nTHINK-BETA reasoning paragraph.",
			"ANSWER-1 first answer paragraph.\n\nANSWER-2 finished tail.",
		);
		component.updateContent(final);
		component.markTranscriptBlockFinalized();
		render();
		const flush = transcript.peekFlushBatch(80);
		expect(flush).toBeDefined();
		transcript.acknowledgeFinalizedBatch(flush!.id);
		const everything = `${streamed}\n${Bun.stripANSI(flush!.rows.join("\n"))}`;

		for (const marker of ["THINK-ALPHA", "THINK-BETA", "ANSWER-1", "ANSWER-2"]) {
			expect(occurrences(everything, marker)).toBe(1);
		}
		// Prose never precedes the thinking that streamed before it.
		expect(everything.indexOf("ANSWER-1")).toBeGreaterThan(everything.indexOf("THINK-BETA"));
	});

	it("stops publishing after a non-append revision and never re-emits published rows", () => {
		const component = new AssistantMessageComponent();
		const transcript = new TranscriptContainer();
		transcript.addChild(component);
		let tick = 0;
		const render = () => transcript.renderViewport(80, 20, { now: tick, tick: tick++ });

		component.updateContent(createAssistantMessage("REV-OLD settled paragraph.\n\nREV-TAIL partial"), {
			transient: true,
		});
		render();
		const first = drainPressureBatches(transcript, 80);
		expect(first.join("\n")).toContain("REV-OLD");
		expect(transcript.emittedStableRows()).toEqual([1]);

		// Legal revision of the already-published paragraph: the container must
		// freeze this block's stable emission rather than publish drifted bytes.
		component.updateContent(
			createAssistantMessage("REV-NEW replacement paragraph.\n\nREV-TAIL partial keeps streaming"),
			{ transient: true },
		);
		render();
		expect(transcript.peekFinalizedBatch(80, 0)).toBeUndefined();

		// Later growth cannot resume emission for the frozen block.
		component.updateContent(
			createAssistantMessage(
				"REV-NEW replacement paragraph.\n\nREV-MIDDLE another settled paragraph.\n\nREV-TAIL partial keeps streaming",
			),
			{ transient: true },
		);
		render();
		expect(transcript.peekFinalizedBatch(80, 0)).toBeUndefined();
		expect(transcript.emittedStableRows()).toEqual([1]);

		component.updateContent(
			createAssistantMessage(
				"REV-NEW replacement paragraph.\n\nREV-MIDDLE another settled paragraph.\n\nREV-TAIL finished.",
			),
		);
		component.markTranscriptBlockFinalized();
		render();
		const flush = transcript.peekFlushBatch(80);
		expect(flush).toBeDefined();
		transcript.acknowledgeFinalizedBatch(flush!.id);
		const flushText = Bun.stripANSI(flush!.rows.join("\n"));
		// The suffix past the emitted row count flushes exactly once; the rows
		// already in native scrollback are never re-emitted.
		expect(occurrences(flushText, "REV-TAIL")).toBe(1);
		expect(flushText).not.toContain("REV-OLD");
	});

	it("holds a growing GFM table in the mutable tail until its block is closed", () => {
		const component = new AssistantMessageComponent();
		const transcript = new TranscriptContainer();
		transcript.addChild(component);
		let tick = 0;
		const render = () => transcript.renderViewport(80, 20, { now: tick, tick: tick++ });

		const tableHead = "| name | value |\n| --- | --- |";
		component.updateContent(
			createAssistantMessage(`TABLE-INTRO paragraph before the table.\n\n${tableHead}\n| row-1 | cell-1 |`),
			{ transient: true },
		);
		render();
		const intro = drainPressureBatches(transcript, 80).join("\n");
		expect(intro).toContain("TABLE-INTRO");
		expect(intro).not.toContain("row-1");

		// Later rows can reflow earlier column widths: the growing table must
		// stay mutable even as it gains rows.
		component.updateContent(
			createAssistantMessage(
				`TABLE-INTRO paragraph before the table.\n\n${tableHead}\n| row-1 | cell-1 |\n| row-2-much-wider-cell | cell-2 |`,
			),
			{ transient: true },
		);
		render();
		expect(drainPressureBatches(transcript, 80)).toEqual([]);

		// A closing blank line plus following content settles the table.
		component.updateContent(
			createAssistantMessage(
				`TABLE-INTRO paragraph before the table.\n\n${tableHead}\n| row-1 | cell-1 |\n| row-2-much-wider-cell | cell-2 |\n\nTABLE-AFTER paragraph partial`,
			),
			{ transient: true },
		);
		render();
		const settled = drainPressureBatches(transcript, 80).join("\n");
		expect(settled).toContain("row-1");
		expect(settled).toContain("row-2-much-wider-cell");

		component.updateContent(
			createAssistantMessage(
				`TABLE-INTRO paragraph before the table.\n\n${tableHead}\n| row-1 | cell-1 |\n| row-2-much-wider-cell | cell-2 |\n\nTABLE-AFTER paragraph done.`,
			),
		);
		component.markTranscriptBlockFinalized();
		render();
		const flush = transcript.peekFlushBatch(80);
		expect(flush).toBeDefined();
		transcript.acknowledgeFinalizedBatch(flush!.id);
		const everything = `${intro}\n${settled}\n${Bun.stripANSI(flush!.rows.join("\n"))}`;
		for (const marker of ["TABLE-INTRO", "row-1", "row-2-much-wider-cell", "TABLE-AFTER"]) {
			expect(occurrences(everything, marker)).toBe(1);
		}
	});

	it("holds an open fenced code block in the mutable tail until it closes", () => {
		const component = new AssistantMessageComponent();
		const transcript = new TranscriptContainer();
		transcript.addChild(component);
		let tick = 0;
		const render = () => transcript.renderViewport(80, 20, { now: tick, tick: tick++ });

		component.updateContent(
			createAssistantMessage("FENCE-INTRO paragraph.\n\n```js\nconst FENCE_BODY_1 = 1;\nconst FENCE_BODY_2 = 2;"),
			{ transient: true },
		);
		render();
		const intro = drainPressureBatches(transcript, 80).join("\n");
		expect(intro).toContain("FENCE-INTRO");
		expect(intro).not.toContain("FENCE_BODY_1");

		component.updateContent(
			createAssistantMessage(
				"FENCE-INTRO paragraph.\n\n```js\nconst FENCE_BODY_1 = 1;\nconst FENCE_BODY_2 = 2;\n```\n\nFENCE-AFTER partial",
			),
			{ transient: true },
		);
		render();
		const settled = drainPressureBatches(transcript, 80).join("\n");
		expect(settled).toContain("FENCE_BODY_1");
		expect(settled).toContain("FENCE_BODY_2");
		expect(settled).not.toContain("FENCE-AFTER");
	});

	it("replays emitted stable prose exactly once at a new width", () => {
		const component = new AssistantMessageComponent();
		const transcript = new TranscriptContainer();
		transcript.addChild(component);
		let tick = 0;
		const render = (width: number) => transcript.renderViewport(width, 20, { now: tick, tick: tick++ });

		component.updateContent(
			createAssistantMessage(
				"WIDE-1 first paragraph long enough to wrap differently at sixty columns.\n\nWIDE-TAIL partial",
			),
			{ transient: true },
		);
		render(80);
		const before = drainPressureBatches(transcript, 80);
		expect(before.join("\n")).toContain("WIDE-1");

		// Settled resize in rebuild mode: native history is cleared and the
		// committed ledger plus the emitted head prefix replays at the new width.
		transcript.beginReplay();
		const replay = transcript.peekReplayBatch(60);
		expect(replay).toBeDefined();
		const replayText = Bun.stripANSI(replay!.rows.join("\n"));
		expect(replayText).toContain("WIDE-1");
		transcript.acknowledgeFinalizedBatch(replay!.id);

		// The stream continues at the new width; only new rows retire.
		component.updateContent(
			createAssistantMessage(
				"WIDE-1 first paragraph long enough to wrap differently at sixty columns.\n\nWIDE-2 second paragraph.\n\nWIDE-TAIL partial grows",
			),
			{ transient: true },
		);
		render(60);
		const after = drainPressureBatches(transcript, 60).join("\n");
		expect(after).toContain("WIDE-2");
		expect(after).not.toContain("WIDE-1");

		component.updateContent(
			createAssistantMessage(
				"WIDE-1 first paragraph long enough to wrap differently at sixty columns.\n\nWIDE-2 second paragraph.\n\nWIDE-TAIL finished.",
			),
		);
		component.markTranscriptBlockFinalized();
		render(60);
		const flush = transcript.peekFlushBatch(60);
		expect(flush).toBeDefined();
		transcript.acknowledgeFinalizedBatch(flush!.id);

		// One post-resize epoch: replay + later batches + flush hold each marker once.
		const epoch = [replayText, after, Bun.stripANSI(flush!.rows.join("\n"))].join("\n");
		for (const marker of ["WIDE-1", "WIDE-2", "WIDE-TAIL"]) {
			expect(occurrences(epoch, marker)).toBe(1);
		}
	});
});
