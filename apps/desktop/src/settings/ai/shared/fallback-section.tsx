import { Trans } from "@lingui/react/macro";
import type { ReactNode } from "react";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@anlg/ui/components/ui/accordion";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@anlg/ui/components/ui/select";
import { cn } from "@anlg/utils";

import { ProviderIconSlot } from "./index";

export const FALLBACK_NONE_VALUE = "none";

export type FallbackProviderOption = {
  id: string;
  displayName: string;
  icon: ReactNode;
  disabled: boolean;
  unconfigured: boolean;
  note?: ReactNode;
};

export function AdvancedFallbackSection({
  description,
  providerValue,
  providerOptions,
  onProviderChange,
  modelPicker,
}: {
  description: ReactNode;
  providerValue: string;
  providerOptions: FallbackProviderOption[];
  onProviderChange: (provider: string) => void;
  modelPicker: ReactNode;
}) {
  return (
    <Accordion type="single" collapsible className="flex flex-col">
      <AccordionItem value="advanced" className="border-none">
        <AccordionTrigger className="text-md font-sans font-semibold hover:no-underline">
          <Trans>Advanced</Trans>
        </AccordionTrigger>
        <AccordionContent className="flex flex-col gap-3">
          <p className="text-muted-foreground text-sm">{description}</p>
          <div className="flex flex-row items-center gap-4">
            <div className="min-w-0 flex-2">
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
            <span className="text-muted-foreground">/</span>
            <div className="min-w-0 flex-3">{modelPicker}</div>
          </div>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}
