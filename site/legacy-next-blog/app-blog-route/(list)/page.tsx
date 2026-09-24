import { Metadata } from "next";
import { getTranslations } from "next-intl/server";

import { BlogCard } from "@/components/blog/blog-card";
import { routing } from "@/i18n/routing";
import { getBlogPosts } from "@/lib/blog";
import { generateBlogJsonLd } from "@/lib/jsonld";
import { constructMetadata } from "@/lib/metadata";
import { jsonldScript } from "@/lib/utils";

type MetadataProps = {
  params: Promise<{ locale: string }>;
};

export async function generateMetadata({
  params,
}: MetadataProps): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale });

  const metadata = await constructMetadata({
    title: t("blog.title"),
    description: t("blogTagline"),
    path: "/blog",
    locale,
  });

  return metadata;
}

export default async function BlogPage(props: {
  params: Promise<{ locale: string }>;
}) {
  const params = await props.params;
  const locale = params.locale || routing.defaultLocale;
  const posts = await getBlogPosts(locale);
  const blogJsonLd = generateBlogJsonLd(posts);
  const t = await getTranslations({ locale });

  return (
    <main className="pt-16 pb-12 sm:pt-24 sm:pb-14 md:pt-32 md:pb-16 lg:pt-36 xl:pt-40">
      {jsonldScript(blogJsonLd)}
      <div className="mx-auto w-full max-w-3xl px-6 sm:px-8 md:px-10">
        <h1 className="mb-4 text-3xl font-semibold tracking-tighter md:text-4xl">
          {t("blog.title")}
        </h1>
        <p className="text-muted-foreground mb-8 max-w-2xl text-sm md:text-base">
          {t("blogTagline")}
        </p>
      </div>

      <div className="mx-auto w-full max-w-3xl px-6 sm:px-8 md:px-10">
        <div className="grid grid-cols-1 gap-4 sm:gap-5">
          {posts
            .sort((a, b) => {
              if (new Date(a.metadata.date) > new Date(b.metadata.date)) {
                return -1;
              }
              return 1;
            })
            .map((post) => (
              <BlogCard
                key={post.slug}
                locale={locale}
                slug={post.slug}
                title={post.metadata.title}
                date={post.metadata.date}
                summary={post.metadata.summary}
              />
            ))}
        </div>
      </div>
    </main>
  );
}
