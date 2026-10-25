import { describe, expect, test } from "vitest";

import { processCheaperInferenceModels } from "./list-cheaper-inference";

describe("processCheaperInferenceModels", () => {
  test("keeps chat models and drops media models by endpoint", () => {
    expect(
      processCheaperInferenceModels([
        {
          id: "gpt-6-luna",
          supported_endpoints: ["/v1/chat/completions", "/v1/responses"],
          capabilities: { vision: true },
        },
        {
          id: "deepseek-v4.1-flash",
          supported_endpoints: ["/v1/chat/completions"],
          capabilities: { vision: false },
        },
        {
          id: "whisper-1",
          supported_endpoints: ["/v1/audio/transcriptions"],
        },
        {
          id: "gpt-4",
          supported_endpoints: ["/v1/chat/completions"],
        },
      ]),
    ).toEqual({
      models: ["deepseek-v4.1-flash", "gpt-6-luna"],
      ignored: [
        { id: "whisper-1", reasons: ["common_keyword", "not_chat_model"] },
        { id: "gpt-4", reasons: ["old_model"] },
      ],
      metadata: {
        "gpt-6-luna": { input_modalities: ["text", "image"] },
        "deepseek-v4.1-flash": { input_modalities: ["text"] },
        "whisper-1": { input_modalities: ["text"] },
        "gpt-4": { input_modalities: ["text"] },
      },
    });
  });
});
