"use client";

import { useLocale, useTranslations } from "next-intl";

import { Icons } from "@/components/icons";
import { Link, LOCALE_TO_HREFLANG } from "@/i18n/routing";

interface BlogCardProps {
  slug: string;
  title: string;
  date: string;
  summary?: string;
  locale?: string;
}

export function BlogCard({
  slug,
  title,
  date,
  summary,
  locale,
}: BlogCardProps) {
  const currentLocale = useLocale();
  const t = useTranslations();
  const displayLocale = locale || currentLocale;

  return (
    <Link
      href={`/blog/${slug}`}
      locale={displayLocale}
      className="group border-border bg-card focus-visible:ring-ring block h-full rounded-xl border p-4 shadow-sm transition-[transform,box-shadow] duration-200 hover:-translate-y-0.5 hover:shadow-md focus-visible:ring-2 focus-visible:outline-none sm:p-5"
    >
      <div className="flex h-full flex-col">
        <div className="mb-3 flex items-center justify-between gap-3">
          <h2 className="line-clamp-2 text-base font-medium tracking-tight md:text-lg">
            {title}
          </h2>
          <span className="bg-secondary text-secondary-foreground group-hover:border-secondary/50 shrink-0 rounded-full border border-transparent px-2.5 py-1 text-[10px] font-medium">
            {new Date(date).toLocaleDateString(
              LOCALE_TO_HREFLANG[
                displayLocale as keyof typeof LOCALE_TO_HREFLANG
              ] || "en-US",
              {
                year: "numeric",
                month: "short",
                day: "2-digit",
              },
            )}
          </span>
        </div>
        {summary ? (
          <p className="text-muted-foreground mb-4 line-clamp-3 text-sm md:mb-5">
            {summary}
          </p>
        ) : null}
        <div className="text-muted-foreground mt-auto flex items-center justify-between pt-2 text-xs">
          <span className="group-hover:text-foreground inline-flex items-center gap-1 transition-colors">
            {t("blog.readMore")}
            <Icons.chevronright className="h-3.5 w-3.5" />
          </span>
        </div>
      </div>
    </Link>
  );
}
