import { Trans } from "@lingui/react/macro";
import type { ReactNode } from "react";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@anlg/ui/components/ui/select";
import { cn } from "@anlg/utils";

import { ProviderIconSlot } from "./index";

import { SettingRow } from "~/settings/setting-row";

export const FALLBACK_NONE_VALUE = "none";

export type FallbackProviderOption = {
  id: string;
  displayName: string;
  icon: ReactNode;
  disabled: boolean;
  unconfigured: boolean;
  note?: ReactNode;
};

export function FallbackRow({
  title,
  description,
  providerValue,
  providerOptions,
  onProviderChange,
  modelPicker,
}: {
  title: ReactNode;
  description: ReactNode;
  providerValue: string;
  providerOptions: FallbackProviderOption[];
  onProviderChange: (provider: string) => void;
  modelPicker: ReactNode;
}) {
  return (
    <SettingRow title={title} description={description} controlWidth="content">
      {(labelProps) => (
        <div
          role="group"
          aria-labelledby={labelProps["aria-labelledby"]}
          aria-describedby={labelProps["aria-describedby"]}
          className="flex w-[26rem] max-w-full items-center gap-2"
        >
          <div className="min-w-0 flex-1">
            <Select value={providerValue} onValueChange={onProviderChange}>
              <SelectTrigger className="bg-card shadow-none">
                <SelectValue placeholder={<Trans>Select a provider</Trans>} />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={FALLBACK_NONE_VALUE}>
                  <Trans>None</Trans>
                </SelectItem>
                {providerOptions.map((provider) => (
                  <SelectItem
                    key={provider.id}
                    value={provider.id}
                    disabled={provider.disabled}
                    className={cn([
                      "data-disabled:text-muted-foreground data-disabled:!opacity-100",
                      provider.unconfigured &&
                        !provider.disabled &&
                        "text-muted-foreground",
                    ])}
                  >
                    <div className="flex flex-col gap-0.5">
                      <div className="flex items-center gap-2">
                        <ProviderIconSlot>{provider.icon}</ProviderIconSlot>
                        <span>{provider.displayName}</span>
                      </div>
                      {provider.note ? (
                        <span className="text-muted-foreground text-[11px]">
                          {provider.note}
                        </span>
                      ) : null}
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="min-w-0 flex-[1.5]">{modelPicker}</div>
        </div>
      )}
    </SettingRow>
  );
}
