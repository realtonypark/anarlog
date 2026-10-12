import { tool } from "ai";
import { z } from "zod";

import { liveQueryClient } from "~/db";
import {
  folderPathMatchesFilter,
  normalizeFolderPath,
} from "~/session/folders";
import { loadFolderPaths } from "~/session/queries/folders";

type ActionItemRow = {
  id: string;
  text: string;
  status: string;
  due_at: string;
  assignee_human_id: string;
  meeting_id: string;
  meeting_title: string;
  folder_path: string;
};

export const buildListActionItemsTool = () =>
  tool({
    description:
      "List action items across every meeting in a folder and its subfolders without reading full notes. Use this for 'all action items in <folder>' questions. Follow next_offset to continue.",
    inputSchema: z.object({
      folder_path: z
        .string()
        .describe(
          "Exact folder path from list_folders or a folder mention, such as CS 101",
        ),
      status: z
        .string()
        .optional()
        .describe("Optional exact status filter, such as todo or done"),
      limit: z.number().int().min(1).max(200).optional(),
      offset: z.number().int().min(0).optional(),
    }),
    execute: async ({ folder_path, status, limit = 100, offset = 0 }) => {
      const path = normalizeFolderPath(folder_path);
      if (!path || !(await loadFolderPaths()).includes(path)) {
        return {
          status: "error" as const,
          message:
            "Folder does not exist. Use list_folders to resolve the exact folder path.",
          folder_path: folder_path,
          total: 0,
          next_offset: null,
          items: [],
        };
      }

      const rows = await liveQueryClient.execute<ActionItemRow>(
        `
          SELECT ai.id, ai.text, ai.status, ai.due_at, ai.assignee_human_id,
                 s.id AS meeting_id, s.title AS meeting_title,
                 s.folder_path AS folder_path
          FROM action_items ai
          JOIN sessions s ON s.id = ai.session_id
          WHERE ai.deleted_at IS NULL
            AND s.deleted_at IS NULL
            AND (s.folder_path = ? OR s.folder_path LIKE ? OR s.folder_path LIKE ?)
          ORDER BY s.created_at DESC, ai.source_order ASC, ai.created_at ASC
        `,
        [path, `${path}/%`, `${path}\\%`],
      );

      const matching = rows.filter(
        (row) =>
          folderPathMatchesFilter(row.folder_path, path) &&
          (status === undefined || row.status === status),
      );
      const items = matching.slice(offset, offset + limit).map((row) => ({
        id: row.id,
        text: row.text,
        status: row.status,
        due_at: row.due_at || null,
        assignee_human_id: row.assignee_human_id || null,
        meeting_id: row.meeting_id,
        meeting_title: row.meeting_title,
      }));

      return {
        status: "ok" as const,
        folder_path: path,
        total: matching.length,
        next_offset:
          offset + items.length < matching.length
            ? offset + items.length
            : null,
        items,
      };
    },
  });
