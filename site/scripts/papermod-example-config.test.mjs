import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import test from "node:test";

const config = readFileSync("hugo-blog/config.yaml", "utf8");
const paperModSvg = readFileSync(
  "hugo-blog/themes/PaperMod/layouts/partials/svg.html",
  "utf8",
);

test("uses PaperMod example-site presentation settings", () => {
  for (const expected of [
    "pygmentsUseClasses: true",
    "mainsections: [\"posts\", \"papermod\"]",
    "pagerSize: 3",
    "ShowShareButtons: true",
    "displayFullLangName: true",
    "ShowRssButtonInSectionTermList: true",
    "ShowAllPagesInArchive: true",
    "ShowPageNums: true",
    "socialIcons:",
  ]) {
    assert.match(config, new RegExp(expected.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
  }
});

test("keeps portfolio navigation in PaperMod's native menu", () => {
  for (const expected of [
    "name: Home",
    'url: "https://chaoyuwang.vercel.app/"',
    "name: Blog",
    "url: /posts/",
    "name: CV",
    'url: "https://chaoyuwang.vercel.app/resume.pdf"',
  ]) {
    assert.match(config, new RegExp(expected.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
  }

  assert.doesNotMatch(config, /name: EN\/中/);
  assert.doesNotMatch(config, /identifier: language/);
});

test("does not override PaperMod's native header partial", () => {
  assert.equal(existsSync("hugo-blog/layouts/partials/header.html"), false);
});

test("orders Research Notes social icons and includes Chinese channels", () => {
  assert.match(
    config,
    /socialIcons:[\s\S]*name: email[\s\S]*name: github[\s\S]*name: linkedin[\s\S]*name: x[\s\S]*name: xiaohongshu[\s\S]*name: zhihu[\s\S]*name: wechat/,
  );
  assert.match(
    config,
    /url: "https:\/\/www\.xiaohongshu\.com\/user\/profile\/58e4dbc982ec39470e251614"/,
  );
  assert.match(config, /url: "\/wechat-qrcode\.jpg"/);
});

test("PaperMod renders the Xiaohongshu social icon", () => {
  assert.match(paperModSvg, /eq \$icon_name "xiaohongshu"/);
});
