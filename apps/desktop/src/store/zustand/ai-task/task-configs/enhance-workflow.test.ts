import type { LanguageModel } from "ai";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import type { TaskArgsMapTransformed } from ".";
import { applyTransforms } from "../shared/transform_infra";
import { enhanceWorkflow } from "./enhance-workflow";

import { formatSummaryDetailGuidance } from "~/services/enhancer/summary-length";

const mocks = vi.hoisted(() => ({ render: vi.fn(), streamText: vi.fn() }));
vi.mock("@anlg/plugin-template", () => ({
  commands: { render: mocks.render },
}));
vi.mock("ai", async (importOriginal) => ({
  ...(await importOriginal<typeof import("ai")>()),
  streamText: mocks.streamText,
}));

beforeEach(() => {
  vi.clearAllMocks();
  mocks.render.mockImplementation(async (input) => ({
    status: "ok",
    data: input.enhanceSystem ? "Rendered system prompt" : "Meeting transcript",
  }));
  mocks.streamText.mockImplementation(() => ({
    fullStream: (async function* () {
      yield {
        type: "text-delta",
        id: "1",
        text: "The team agreed to ship Friday.",
      };
    })(),
  }));
});

it("passes custom formatting to the renderer without adding layout rules", async () => {
  const formatOverride =
    "Write an executive overview in prose, followed by discussion prose and bullets.";
  const args: TaskArgsMapTransformed["enhance"] = {
    language: "en",
    formatOverride,
    session: { title: "Launch", startedAt: null, endedAt: null, event: null },
    participants: [],
    template: null,
    preMeetingMemo: "",
    postMeetingMemo: "",
    transcripts: [
      {
        startedAt: null,
        endedAt: null,
        nowMs: null,
        segments: [{ speaker: "John", text: "a".repeat(636), startMs: null }],
      },
    ],
    imageContext: [],
    lengthPolicy: {
      transcript_characters: 636,
      guidance: {
        max_characters: 636,
        min_sections: 1,
        max_sections: 2,
      },
    },
    dictionaryTerms: [],
  };
  const chunks = [];
  for await (const chunk of enhanceWorkflow.executeWorkflow!({
    model: { provider: "openai.responses" } as LanguageModel,
    args,
    onProgress: vi.fn(),
    signal: new AbortController().signal,
  }))
    chunks.push(chunk);
  expect(chunks).toHaveLength(1);
  expect(mocks.streamText).toHaveBeenCalledTimes(1);
  const request = mocks.streamText.mock.calls[0][0];
  expect(mocks.render).toHaveBeenCalledWith({
    enhanceSystem: { language: "en", formatOverride },
  });
  expect(request.system).toContain("Rendered system prompt");
  expect(request.system).toContain(formatSummaryDetailGuidance(false));
  expect(request.system).not.toMatch(
    /never put prose|bullets per section|# Next Steps/,
  );
  expect(request.prompt).toContain("Keep the requested structure");
  expect(request.prompt).not.toMatch(/\d to \d sections/);
  expect(request.maxOutputTokens).toBeUndefined();
});

it("adds length guidance to prompts rendered from a template with sections", async () => {
  const args: TaskArgsMapTransformed["enhance"] = {
    language: "en",
    formatOverride: "",
    session: { title: "Launch", startedAt: null, endedAt: null, event: null },
    participants: [],
    template: {
      title: "1:1 Meeting",
      description: null,
      sections: [
        { title: "Updates", description: "" },
        { title: "Next Steps", description: "" },
      ],
    },
    preMeetingMemo: "",
    postMeetingMemo: "",
    transcripts: [
      {
        startedAt: null,
        endedAt: null,
        nowMs: null,
        segments: [
          { speaker: "John", text: "a".repeat(10_000), startMs: null },
        ],
      },
    ],
    imageContext: [],
    lengthPolicy: {
      transcript_characters: 10_000,
      guidance: {
        max_characters: 10_000,
        min_sections: 3,
        max_sections: 6,
      },
    },
    dictionaryTerms: [],
  };
  for await (const _ of enhanceWorkflow.executeWorkflow!({
    model: { provider: "anthropic.messages" } as LanguageModel,
    args,
    onProgress: vi.fn(),
    signal: new AbortController().signal,
  })) {
    // drain the stream
  }
  const request = mocks.streamText.mock.calls[0][0];
  expect(request.prompt).toContain("Summary length:");
  expect(request.prompt).toContain("Keep every requested template section");
  expect(request.prompt).not.toMatch(/\d to \d sections/);
  expect(request.maxOutputTokens).toBe(64_000);
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllEnvs();
});

it.each([
  { rollback: undefined, elapsed: 0 },
  { rollback: "true", elapsed: 500 },
])(
  "preserves normalized summary bytes without a streaming backlog (rollback=$rollback)",
  async ({ rollback, elapsed }) => {
    vi.useFakeTimers();
    vi.stubEnv("VITE_DISABLE_MEETING_PERF_OPTIMIZATIONS", rollback);
    vi.resetModules();
    const { enhanceWorkflow: workflow } = await import("./enhance-workflow");
    const stream = (async function* () {
      yield { type: "text-start" as const, id: "summary" };
      yield {
        type: "text-delta" as const,
        id: "summary",
        text: "• First\n• Sec",
      };
      yield { type: "text-delta" as const, id: "summary", text: "ond\n- Tail" };
      yield { type: "text-end" as const, id: "summary" };
    })();
    const startedAt = Date.now();
    const completion = (async () => {
      let text = "";
      for await (const chunk of applyTransforms(
        stream,
        workflow.transforms ?? [],
      )) {
        if (chunk.type === "text-delta") text += chunk.text;
      }
      return { text, elapsed: Date.now() - startedAt };
    })();
    await vi.runAllTimersAsync();
    expect(await completion).toEqual({
      text: "- First\n- Second\n- Tail",
      elapsed,
    });
  },
);
