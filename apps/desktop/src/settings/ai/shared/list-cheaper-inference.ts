import { Effect, pipe, Schema } from "effect";

import {
  DEFAULT_RESULT,
  extractMetadataMap,
  fetchJson,
  type InputModality,
  isNonChatModel,
  isOldModel,
  type ListModelsResult,
  type ModelIgnoreReason,
  partition,
  REQUEST_TIMEOUT,
  shouldIgnoreCommonKeywords,
  sortModelsByRecency,
} from "./list-common";

const CheaperInferenceModelSchema = Schema.Struct({
  data: Schema.Array(
    Schema.Struct({
      id: Schema.String,
      supported_endpoints: Schema.optional(Schema.Array(Schema.String)),
      capabilities: Schema.optional(
        Schema.Struct({
          vision: Schema.optional(Schema.Boolean),
        }),
      ),
    }),
  ),
});

type CheaperInferenceModel = Schema.Schema.Type<
  typeof CheaperInferenceModelSchema
>["data"][number];

export async function listCheaperInferenceModels(
  baseUrl: string,
  apiKey: string,
): Promise<ListModelsResult> {
  if (!baseUrl) {
    return DEFAULT_RESULT;
  }

  return pipe(
    fetchJson(`${baseUrl}/models`, { Authorization: `Bearer ${apiKey}` }),
    Effect.andThen((json) =>
      Schema.decodeUnknown(CheaperInferenceModelSchema)(json),
    ),
    Effect.map(({ data }) => processCheaperInferenceModels(data)),
    Effect.timeout(REQUEST_TIMEOUT),
    Effect.catchAll(() => Effect.succeed(DEFAULT_RESULT)),
    Effect.runPromise,
  );
}

export const processCheaperInferenceModels = (
  data: ReadonlyArray<CheaperInferenceModel>,
): ListModelsResult => {
  const result = partition(data, getIgnoreReasons, (model) => model.id);

  return {
    models: sortModelsByRecency(result.models),
    ignored: result.ignored,
    metadata: extractMetadataMap(
      data,
      (model) => model.id,
      (model) => ({ input_modalities: getInputModalities(model) }),
    ),
  };
};

const supportsChatCompletions = (model: CheaperInferenceModel): boolean =>
  !Array.isArray(model.supported_endpoints) ||
  model.supported_endpoints.includes("/v1/chat/completions");

const getIgnoreReasons = (
  model: CheaperInferenceModel,
): ModelIgnoreReason[] | null => {
  const reasons: ModelIgnoreReason[] = [];
  if (shouldIgnoreCommonKeywords(model.id)) {
    reasons.push("common_keyword");
  }
  if (!supportsChatCompletions(model)) {
    reasons.push("not_chat_model");
  } else if (isNonChatModel(model.id)) {
    reasons.push("not_chat_model");
  }
  if (isOldModel(model.id)) {
    reasons.push("old_model");
  }
  return reasons.length > 0 ? reasons : null;
};

const getInputModalities = (model: CheaperInferenceModel): InputModality[] => [
  "text",
  ...(model.capabilities?.vision ? (["image"] as const) : []),
];
