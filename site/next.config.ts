import type { NextConfig } from "next";
import createNextIntlPlugin from "next-intl/plugin";

const withNextIntl = createNextIntlPlugin("./src/i18n/request.ts");

const nextConfig: NextConfig = {
  reactStrictMode: true,
  async rewrites() {
    const blogRewrites = [
      {
        source: "/blog",
        destination: "/blog/index.html",
      },
      {
        source: "/blog/posts",
        destination: "/blog/posts/index.html",
      },
      {
        source: "/blog/posts/:slug",
        destination: "/blog/posts/:slug/index.html",
      },
      {
        source: "/blog/archives",
        destination: "/blog/archives/index.html",
      },
      {
        source: "/blog/search",
        destination: "/blog/search/index.html",
      },
      {
        source: "/blog/tags",
        destination: "/blog/tags/index.html",
      },
      {
        source: "/blog/categories",
        destination: "/blog/categories/index.html",
      },
    ];
    const localizedBlogRewrites = blogRewrites.map((rewrite) => ({
      source: `/zh${rewrite.source}`,
      destination: rewrite.destination,
    }));

    return [
      ...blogRewrites,
      ...localizedBlogRewrites,
      {
        source: "/open-source/nextjs-portfolio-blog-research/docs",
        destination:
          "https://nextjs-portfolio-blog-research-docs.vercel.app/open-source/nextjs-portfolio-blog-research/docs",
      },
      {
        source: "/open-source/nextjs-portfolio-blog-research/docs/:path*",
        destination:
          "https://nextjs-portfolio-blog-research-docs.vercel.app/open-source/nextjs-portfolio-blog-research/docs/:path*",
      },
    ];
  },
};

export default withNextIntl(nextConfig);
