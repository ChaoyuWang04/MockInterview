import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import test from "node:test";

const postsDir = "hugo-blog/content/posts";

function parseFrontMatter(markdown) {
  const match = markdown.match(/^---\n([\s\S]*?)\n---\n/);
  if (!match) {
    return null;
  }

  return match[1];
}

test("Hugo posts include front matter needed by PaperMod list cards", () => {
  const postFiles = readdirSync(postsDir).filter((name) => name.endsWith(".md"));

  for (const fileName of postFiles) {
    const frontMatter = parseFrontMatter(
      readFileSync(`${postsDir}/${fileName}`, "utf8"),
    );

    assert.ok(frontMatter, `${fileName} is missing front matter`);
    assert.match(frontMatter, /^title:\s*.+$/m, `${fileName} is missing title`);

    if (!/^draft:\s*true$/m.test(frontMatter)) {
      assert.match(frontMatter, /^date:\s*.+$/m, `${fileName} is missing date`);
    }
  }
});
