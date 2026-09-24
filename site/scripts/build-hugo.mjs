import { execFileSync } from "node:child_process";
import { createWriteStream, existsSync } from "node:fs";
import { chmod, mkdir } from "node:fs/promises";
import https from "node:https";
import { arch, platform } from "node:os";
import path from "node:path";
import { pipeline } from "node:stream/promises";
import { fileURLToPath } from "node:url";

const DEFAULT_HUGO_VERSION = "0.160.1";

export function getHugoAssetName({
  platform: targetPlatform = platform(),
  arch: targetArch = arch(),
  version = DEFAULT_HUGO_VERSION,
} = {}) {
  const platformNames = {
    linux: "linux",
  };
  const archNames = {
    x64: "amd64",
    arm64: "arm64",
  };

  const releasePlatform = platformNames[targetPlatform];
  const releaseArch = archNames[targetArch];

  if (!releasePlatform || !releaseArch) {
    throw new Error(
      `Unsupported Hugo platform: ${targetPlatform}/${targetArch}`,
    );
  }

  return `hugo_extended_withdeploy_${version}_${releasePlatform}-${releaseArch}.tar.gz`;
}

export function getHugoDownloadUrl(options = {}) {
  const version = options.version ?? DEFAULT_HUGO_VERSION;
  const assetName = getHugoAssetName({ ...options, version });

  return `https://github.com/gohugoio/hugo/releases/download/v${version}/${assetName}`;
}

async function downloadFile(url, destination, redirectCount = 0) {
  if (redirectCount > 5) {
    throw new Error(`Too many redirects downloading ${url}`);
  }

  await new Promise((resolve, reject) => {
    https
      .get(url, (response) => {
        const statusCode = response.statusCode ?? 0;

        if (
          statusCode >= 300 &&
          statusCode < 400 &&
          response.headers.location
        ) {
          response.resume();
          const redirectUrl = new URL(response.headers.location, url).toString();
          downloadFile(redirectUrl, destination, redirectCount + 1)
            .then(resolve)
            .catch(reject);
          return;
        }

        if (statusCode < 200 || statusCode >= 300) {
          response.resume();
          reject(new Error(`Download failed with HTTP ${statusCode}: ${url}`));
          return;
        }

        const fileStream = createWriteStream(destination);
        pipeline(response, fileStream).then(resolve).catch(reject);
      })
      .on("error", reject);
  });
}

async function ensureHugoBinary() {
  if (process.env.HUGO_BIN) {
    return process.env.HUGO_BIN;
  }

  if (platform() !== "linux") {
    try {
      execFileSync("hugo", ["version"], { stdio: "ignore" });
      return "hugo";
    } catch {
      throw new Error(
        "Set HUGO_BIN to a local Hugo executable on non-Linux platforms.",
      );
    }
  }

  const version = process.env.HUGO_VERSION ?? DEFAULT_HUGO_VERSION;
  const cacheRoot =
    process.env.HUGO_CACHE_DIR ??
    path.join(process.cwd(), "node_modules", ".cache", "hugo");
  const cacheDir = path.join(cacheRoot, `v${version}-${platform()}-${arch()}`);
  const hugoBin = path.join(cacheDir, "hugo");

  if (existsSync(hugoBin)) {
    return hugoBin;
  }

  await mkdir(cacheDir, { recursive: true });

  const assetName = getHugoAssetName({ version });
  const archivePath = path.join(cacheDir, assetName);
  const downloadUrl = getHugoDownloadUrl({ version });

  console.log(`Downloading Hugo ${version} from ${downloadUrl}`);
  await downloadFile(downloadUrl, archivePath);

  execFileSync("tar", ["-xzf", archivePath, "-C", cacheDir], {
    stdio: "inherit",
  });
  await chmod(hugoBin, 0o755);

  return hugoBin;
}

export async function buildHugo() {
  const hugoBin = await ensureHugoBinary();
  execFileSync(hugoBin, ["--source", "hugo-blog", "--minify"], {
    stdio: "inherit",
  });
}

const currentFile = fileURLToPath(import.meta.url);

if (process.argv[1] === currentFile) {
  buildHugo().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
