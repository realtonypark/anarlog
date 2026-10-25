import { beforeEach, describe, expect, test, vi } from "vitest";

import {
  listCheaperInferenceModels,
  processCheaperInferenceModels,
} from "./list-cheaper-inference";

import { providerFetch } from "~/ai/provider-fetch";

vi.mock("~/ai/provider-fetch", () => ({ providerFetch: vi.fn() }));

beforeEach(() => vi.resetAllMocks());

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

describe("listCheaperInferenceModels", () => {
  test("fetches the catalog with Bearer auth and keeps chat-capable models", async () => {
    vi.mocked(providerFetch).mockResolvedValue(
      Response.json({
        data: [
          {
            id: "gpt-6-luna",
            supported_endpoints: ["/v1/chat/completions", "/v1/responses"],
            capabilities: { vision: true },
          },
          {
            id: "whisper-1",
            supported_endpoints: ["/v1/audio/transcriptions"],
          },
        ],
      }),
    );
    const result = await listCheaperInferenceModels(
      "https://api.cheaperinference.com/v1",
      "test-key",
    );
    expect(providerFetch).toHaveBeenCalledWith(
      "https://api.cheaperinference.com/v1/models",
      {
        method: "GET",
        headers: { Authorization: "Bearer test-key" },
      },
    );
    expect(result.models).toEqual(["gpt-6-luna"]);
    expect(result.metadata).toEqual({
      "gpt-6-luna": { input_modalities: ["text", "image"] },
      "whisper-1": { input_modalities: ["text"] },
    });
  });

  test("returns no models when the catalog request fails", async () => {
    vi.mocked(providerFetch).mockResolvedValue(
      new Response("unavailable", { status: 503 }),
    );
    expect(
      await listCheaperInferenceModels(
        "https://api.cheaperinference.com/v1",
        "test-key",
      ),
    ).toEqual({ models: [], ignored: [], metadata: {} });
  });
});
