"use client";

import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";

import { cn } from "@/lib/utils";

interface TOCItem {
  id: string;
  text: string;
  level: number;
}

interface TableOfContentsProps {
  content: string;
  className?: string;
  hideTitle?: boolean;
  onItemClick?: () => void;
}

export function TableOfContents({
  content,
  className,
  hideTitle = false,
  onItemClick,
}: TableOfContentsProps) {
  const [toc, setToc] = useState<TOCItem[]>([]);
  const t = useTranslations();

  // Extract headings from HTML content
  useEffect(() => {
    const parser = new DOMParser();
    const doc = parser.parseFromString(content, "text/html");
    const headings = doc.querySelectorAll("h1, h2, h3, h4, h5, h6");

    const tocItems: TOCItem[] = Array.from(headings)
      .filter((heading) => heading.id) // Only include headings with IDs
      .map((heading) => {
        const id = heading.id;
        const text = heading.textContent?.trim() || "";
        const level = parseInt(heading.tagName.charAt(1));

        return { id, text, level };
      })
      .filter((item) => item.text.length > 0); // Filter out empty headings
    setToc(tocItems);
  }, [content]);

  const scrollToHeading = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    // Call the callback to close mobile TOC if provided
    if (onItemClick) {
      onItemClick();
    }
  };

  const tocContent = (
    <>
      {!hideTitle && (
        <div className="mb-4">
          <h3 className="text-foreground/90 text-sm font-medium tracking-wide">
            {t("blog.toc.title")}
          </h3>
        </div>
      )}
      {toc.length === 0 ? (
        <div className="text-muted-foreground/70 py-8 text-center text-sm">
          <p>{t("blog.toc.empty")}</p>
        </div>
      ) : (
        <nav className="space-y-0.5">
          {toc.map((item) => (
            <button
              key={item.id}
              onClick={() => scrollToHeading(item.id)}
              className={cn(
                "w-full text-left text-sm transition-colors duration-200",
                "hover:bg-muted/50 -mx-3 -my-0.5 rounded-lg px-3 py-2",
                "text-muted-foreground hover:text-foreground/80",
                {
                  "pl-3": item.level === 1,
                  "pl-6": item.level === 2,
                  "pl-9": item.level === 3,
                  "pl-12": item.level === 4,
                  "pl-15": item.level === 5,
                  "pl-18": item.level === 6,
                },
              )}
            >
              <span className="block truncate leading-relaxed">
                {item.text}
              </span>
            </button>
          ))}
        </nav>
      )}
    </>
  );

  if (hideTitle) {
    return <div className={cn("w-full", className)}>{tocContent}</div>;
  }

  return (
    <div className={cn("w-62", className)}>
      <div className="border-border/50 bg-background/80 rounded-xl border p-5 shadow-2xl shadow-black/5 backdrop-blur-xl dark:shadow-black/20">
        {tocContent}
      </div>
    </div>
  );
}
