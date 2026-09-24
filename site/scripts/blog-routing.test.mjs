import assert from "node:assert/strict";
import test from "node:test";

import { createJiti } from "jiti";

const jiti = createJiti(import.meta.url);
const nextConfig = await jiti.import("../next.config.ts", { default: true });
const rewrites = await nextConfig.rewrites();

test("rewrites Chinese-prefixed blog routes to the Hugo subsite output", () => {
  assert.deepEqual(
    rewrites
      .filter((rewrite) => rewrite.source.startsWith("/zh/blog"))
      .map(({ source, destination }) => ({ source, destination })),
    [
      {
        source: "/zh/blog",
        destination: "/blog/index.html",
      },
      {
        source: "/zh/blog/posts",
        destination: "/blog/posts/index.html",
      },
      {
        source: "/zh/blog/posts/:slug",
        destination: "/blog/posts/:slug/index.html",
      },
      {
        source: "/zh/blog/archives",
        destination: "/blog/archives/index.html",
      },
      {
        source: "/zh/blog/search",
        destination: "/blog/search/index.html",
      },
      {
        source: "/zh/blog/tags",
        destination: "/blog/tags/index.html",
      },
      {
        source: "/zh/blog/categories",
        destination: "/blog/categories/index.html",
      },
    ],
  );
});
