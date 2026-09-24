import { getTranslations } from "next-intl/server";

import { siteConfig } from "@/data/site";
import {
  DEFAULT_LOCALE,
  getLocaleUrl,
  type Locale,
  routing,
} from "@/i18n/routing";
import { getBlogPosts } from "@/lib/blog";

// ISR configuration - revalidate every hour (3600 seconds)
export const revalidate = 3600;

const SUPPORTED_LOCALES = routing.locales;
const DEFAULT_FEED_LOCALE = DEFAULT_LOCALE as Locale;

function isLocale(value: string | null): value is Locale {
  if (!value) {
    return false;
  }
  return SUPPORTED_LOCALES.includes(value as Locale);
}

function getLocaleFromAcceptLanguage(value: string | null): Locale | null {
  if (!value) {
    return null;
  }

  // Minimal Accept-Language parser:
  // "fr-CH, fr;q=0.9, en;q=0.8" -> ["fr-ch","fr","en"]
  const candidates = value
    .split(",")
    .map((part) => part.trim().split(";")[0]?.toLowerCase())
    .filter(Boolean) as string[];

  const supportedLower = new Map<string, Locale>(
    SUPPORTED_LOCALES.map((l) => [l.toLowerCase(), l]),
  );

  for (const candidate of candidates) {
    // Try exact match first (for locales like "pt-br" if you ever add them)
    const exact = supportedLower.get(candidate);
    if (exact) {
      return exact;
    }

    // Then try primary subtag: "fr-ch" -> "fr"
    const primary = candidate.split("-")[0];
    if (!primary) {
      continue;
    }
    const primaryMatch = supportedLower.get(primary);
    if (primaryMatch) {
      return primaryMatch;
    }
  }

  return null;
}

function resolveLocale(request: Request): {
  locale: Locale;
  explicit: boolean;
} {
  const url = new URL(request.url);
  const fromQuery =
    url.searchParams.get("locale") ?? url.searchParams.get("lang");
  if (isLocale(fromQuery)) {
    return { locale: fromQuery, explicit: true };
  }

  const fromHeader = getLocaleFromAcceptLanguage(
    request.headers.get("accept-language"),
  );
  return { locale: fromHeader ?? DEFAULT_FEED_LOCALE, explicit: false };
}

function getSelfFeedUrl(locale: Locale, explicit: boolean): string {
  const u = new URL("/api/feed/atom.xml", siteConfig.url);
  // Keep backward-compat for the default locale unless explicitly requested.
  if (explicit || locale !== DEFAULT_FEED_LOCALE) {
    u.searchParams.set("locale", locale);
  }
  return u.toString();
}

export async function GET(request: Request) {
  const { locale, explicit } = resolveLocale(request);
  const posts = await getBlogPosts(locale);
  const t = await getTranslations({ locale });
  const socialData = t.raw("social") as Record<
    string,
    {
      name: string;
      url: string;
      icon: string;
      navbar?: boolean;
      content?: boolean;
      footer?: boolean;
    }
  >;

  // Sort posts by published date (newest first)
  const sortedPosts = posts.sort((a, b) => {
    return (
      new Date(b.metadata.date).getTime() - new Date(a.metadata.date).getTime()
    );
  });

  const nameFull = t("name.full");
  const blogTitle = t("blog.title");
  const headline = t("headline").replace(/\n/g, ", ");
  const blogUrl = getLocaleUrl(locale, "/blog");
  const selfFeedUrl = getSelfFeedUrl(locale, explicit);
  const feedTitle = `${blogTitle} | ${nameFull}`;
  const atomFeed = `<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xml:lang="${locale}">
   <title>${escapeXml(feedTitle)}</title>
  <subtitle>${escapeXml(headline)}</subtitle>
  <link href="${selfFeedUrl}" rel="self"/>
  <link href="${blogUrl}"/>
  <id>${blogUrl}</id>
  <author>
    <name>${escapeXml(nameFull)}</name>
    <email>${extractEmailAddress(socialData.email.url)}</email>
  </author>
  <updated>${new Date().toISOString()}</updated>
  ${sortedPosts
    .map((post) => {
      const postUrl = getLocaleUrl(locale, `/blog/${post.slug}`);
      const entryTitle = `${post.metadata.title} | ${nameFull}`;
      const publishedDate = new Date(post.metadata.date).toISOString();
      const updatedDate = new Date(post.metadata.date).toISOString();

      return `
  <entry>
    <title>${escapeXml(entryTitle)}</title>
    <link href="${postUrl}"/>
    <id>${postUrl}</id>
    <published>${publishedDate}</published>
    <updated>${updatedDate}</updated>
    <author>
      <name>${escapeXml(nameFull)}</name>
      <email>${extractEmailAddress(socialData.email.url)}</email>
    </author>
    <summary>${escapeXml(post.metadata.summary || "")}</summary>
    <content type="html">${escapeXml(post.source)}</content>
  </entry>`;
    })
    .join("")}
</feed>`;

  return new Response(atomFeed, {
    headers: {
      "Content-Type": "application/atom+xml; charset=utf-8",
      "Cache-Control": "public, max-age=3600, s-maxage=3600",
      "Content-Language": locale,
      Vary: "Accept-Language",
    },
  });
}

function escapeXml(unsafe: string): string {
  return unsafe
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function extractEmailAddress(emailUrl: string): string {
  // Remove mailto: prefix if present
  return emailUrl.replace(/^mailto:/i, "");
}
