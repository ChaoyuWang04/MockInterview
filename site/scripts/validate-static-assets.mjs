import fs from "node:fs";
import path from "node:path";

const PROJECT_ROOT = process.cwd();
const SOURCE_DIRS = ["src", "content"];
const SOURCE_EXTENSIONS = new Set([
  ".ts",
  ".tsx",
  ".js",
  ".jsx",
  ".mjs",
  ".cjs",
  ".json",
  ".md",
  ".mdx",
  ".css",
]);
const STATIC_ASSET_RE =
  /(^|[\s"'`(=:>])(?<asset>\/[A-Za-z0-9\-._~%!$&'()*+,;=:@/]+?\.(?:png|jpe?g|gif|webp|svg|ico|pdf|mp4|webm|mov))(?=$|[\s"'`)>,?])/g;
const NEXT_METADATA_FILES = [
  "src/app/favicon.ico",
  "src/app/icon.png",
  "src/app/apple-icon.png",
];

function walk(dirPath) {
  if (!fs.existsSync(dirPath)) {
    return [];
  }

  const entries = fs.readdirSync(dirPath, { withFileTypes: true });
  const files = [];

  for (const entry of entries) {
    const fullPath = path.join(dirPath, entry.name);

    if (entry.isDirectory()) {
      files.push(...walk(fullPath));
      continue;
    }

    if (SOURCE_EXTENSIONS.has(path.extname(entry.name))) {
      files.push(fullPath);
    }
  }

  return files;
}

function hasExactCasePath(relativePath) {
  const segments = relativePath.split("/").filter(Boolean);
  let currentDir = PROJECT_ROOT;

  for (const segment of segments) {
    if (!fs.existsSync(currentDir) || !fs.statSync(currentDir).isDirectory()) {
      return false;
    }

    const entries = new Set(fs.readdirSync(currentDir));
    if (!entries.has(segment)) {
      return false;
    }

    currentDir = path.join(currentDir, segment);
  }

  return fs.existsSync(currentDir);
}

function normalizeAssetPath(assetPath) {
  const [pathname] = assetPath.split(/[?#]/, 1);
  return pathname;
}

function isServedByNextMetadata(assetPath) {
  const basename = path.posix.basename(assetPath);

  return NEXT_METADATA_FILES.some((filePath) => {
    return (
      basename === path.posix.basename(filePath) &&
      hasExactCasePath(filePath)
    );
  });
}

const references = new Map();

for (const sourceDir of SOURCE_DIRS) {
  for (const filePath of walk(path.join(PROJECT_ROOT, sourceDir))) {
    const content = fs.readFileSync(filePath, "utf8");

    for (const match of content.matchAll(STATIC_ASSET_RE)) {
      const assetPath = match.groups?.asset;
      if (!assetPath || assetPath.startsWith("/_next/") || assetPath.startsWith("/_vercel/")) {
        continue;
      }

      const normalizedPath = normalizeAssetPath(assetPath);
      const publicRelativePath = path.posix.join("public", normalizedPath.slice(1));
      const exactExists = hasExactCasePath(publicRelativePath);

      if (exactExists || isServedByNextMetadata(normalizedPath)) {
        continue;
      }

      const sourceRelativePath = path.relative(PROJECT_ROOT, filePath);
      const foundIn = references.get(normalizedPath) ?? [];
      foundIn.push(sourceRelativePath);
      references.set(normalizedPath, foundIn);
    }
  }
}

if (references.size > 0) {
  console.error("Static asset validation failed. These public asset references are missing or use the wrong filename case:");

  for (const [assetPath, filePaths] of [...references.entries()].sort(([a], [b]) => a.localeCompare(b))) {
    console.error(`- ${assetPath}`);
    for (const filePath of filePaths) {
      console.error(`  referenced from ${filePath}`);
    }
  }

  console.error("Fix the path or rename the file in public/ so the casing matches exactly.");
  process.exit(1);
}

console.log("Static asset validation passed.");
