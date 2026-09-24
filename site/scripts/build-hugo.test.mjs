import assert from "node:assert/strict";
import test from "node:test";

import { getHugoAssetName, getHugoDownloadUrl } from "./build-hugo.mjs";

test("resolves the Hugo extended release asset for Vercel linux x64", () => {
  assert.equal(
    getHugoAssetName({
      platform: "linux",
      arch: "x64",
      version: "0.160.1",
    }),
    "hugo_extended_withdeploy_0.160.1_linux-amd64.tar.gz",
  );
});

test("builds the GitHub release download URL for a fixed Hugo version", () => {
  assert.equal(
    getHugoDownloadUrl({
      platform: "linux",
      arch: "x64",
      version: "0.160.1",
    }),
    "https://github.com/gohugoio/hugo/releases/download/v0.160.1/hugo_extended_withdeploy_0.160.1_linux-amd64.tar.gz",
  );
});
