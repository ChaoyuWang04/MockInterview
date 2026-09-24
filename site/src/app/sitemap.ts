import { MetadataRoute } from "next";

import { siteConfig } from "@/data/site";
import { DEFAULT_LOCALE, LOCALES } from "@/i18n/routing";
import { getBlogPosts } from "@/lib/blog";

const siteUrl = siteConfig.url;

type ChangeFrequency =
  | "always"
  | "hourly"
  | "daily"
  | "weekly"
  | "monthly"
  | "yearly"
  | "never"
  | undefined;

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  // Static pages
  const staticPages = ["", "/blog", "/privacy-policy", "/terms-of-service"];

  const pages = LOCALES.flatMap((locale) => {
    return staticPages.map((page) => ({
      url: `${siteUrl}${locale === DEFAULT_LOCALE ? "" : `/${locale}`}${page}`,
      lastModified: new Date(),
      changeFrequency: (["", "/blog"].includes(page)
        ? "weekly"
        : "monthly") as ChangeFrequency,
      priority: page === "" ? 1.0 : page === "/blog" ? 0.8 : 0.5,
    }));
  });

  const allBlogSitemapEntries: MetadataRoute.Sitemap = [];

  for (const locale of LOCALES) {
    const posts = await getBlogPosts(locale);
    posts
      .filter(
        (post) =>
          post.slug &&
          (post.metadata.status !== "draft" || !post.metadata.status),
      )
      .forEach((post) => {
        const slugPart = post.slug.replace(/^\//, "").replace(/^blogs\//, "");
        if (slugPart) {
          allBlogSitemapEntries.push({
            url: `${siteUrl}${
              locale === DEFAULT_LOCALE ? "" : `/${locale}`
            }/blog/${slugPart}`,
            lastModified: post.metadata.updatedAt
              ? new Date(post.metadata.updatedAt as string)
              : post.metadata.date
                ? new Date(post.metadata.date)
                : new Date(),
            changeFrequency: "monthly" as ChangeFrequency,
            priority: 0.7,
          });
        }
      });
  }

  const uniqueBlogPostEntries = Array.from(
    new Map(allBlogSitemapEntries.map((entry) => [entry.url, entry])).values(),
  );

  return [...pages, ...uniqueBlogPostEntries];
}
