"use client";

import { useTranslations } from "next-intl";
import React, { useState } from "react";

import { cn } from "@/lib/utils";

import { Button } from "../../ui/button";
import { TableOfContents } from "./table-of-contents";

interface MobileTOCProps {
  content: string;
}

export function MobileTOC({ content }: MobileTOCProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [hasHeadings, setHasHeadings] = useState(false);
  const t = useTranslations();

  // Check if content has headings
  React.useEffect(() => {
    const parser = new DOMParser();
    const doc = parser.parseFromString(content, "text/html");
    const headings = doc.querySelectorAll("h1, h2, h3, h4, h5, h6");
    setHasHeadings(headings.length > 0);
  }, [content]);

  // Don't render if no headings
  if (!hasHeadings) {
    return null;
  }

  return (
    <>
      {/* Floating TOC Button - Only visible on mobile and tablet */}
      <div className="fixed right-6 bottom-6 z-50 xl:hidden">
        <Button
          onClick={() => setIsOpen(!isOpen)}
          size="sm"
          className="bg-background/90 border-border/50 hover:bg-background/95 text-foreground h-12 w-12 rounded-full border shadow-xl shadow-black/10 backdrop-blur-xl dark:shadow-black/30"
          aria-label={t("blog.toc.toggle")}
        >
          <svg
            className="h-5 w-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 6h16M4 12h16M4 18h16"
            />
          </svg>
        </Button>
      </div>

      {/* Mobile TOC Overlay */}
      <div
        className={cn(
          "fixed inset-0 z-40 bg-black/50 transition-opacity xl:hidden",
          isOpen ? "opacity-100" : "pointer-events-none opacity-0",
        )}
        onClick={() => setIsOpen(false)}
      >
        <div
          className={cn(
            "bg-background/95 border-border/50 fixed top-0 right-0 h-full w-80 max-w-[85vw] transform border-l shadow-2xl shadow-black/20 backdrop-blur-xl transition-transform xl:hidden",
            isOpen ? "translate-x-0" : "translate-x-full",
          )}
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex h-full flex-col">
            {/* Header */}
            <div className="border-border/50 flex items-center justify-between border-b p-4">
              <h2 className="text-foreground/90 text-lg font-medium tracking-wide">
                {t("blog.toc.title")}
              </h2>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsOpen(false)}
                className="text-foreground h-8 w-8 p-0"
                aria-label={t("blog.toc.close")}
              >
                <svg
                  className="h-4 w-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </Button>
            </div>

            {/* TOC Content */}
            <div className="flex-1 p-4">
              <TableOfContents
                content={content}
                hideTitle={true}
                onItemClick={() => setIsOpen(false)}
              />
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
