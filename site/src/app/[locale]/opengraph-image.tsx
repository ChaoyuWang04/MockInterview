import { readFile } from "node:fs/promises";
import { join } from "node:path";

import { ImageResponse } from "next/og";
import { getTranslations } from "next-intl/server";

import { siteConfig } from "@/data/site";

// Helper function to load Google Font dynamically
async function loadGoogleFont(
  font: string,
  weight: number,
  text?: string,
): Promise<ArrayBuffer> {
  const weightParam = `wght@${weight}`;
  const textParam = text ? `&text=${encodeURIComponent(text)}` : "";
  const url = `https://fonts.googleapis.com/css2?family=${font}:${weightParam}${textParam}`;
  const css = await (await fetch(url)).text();
  const resource = css.match(
    /src: url\((.+)\) format\('(opentype|truetype|woff2)'\)/,
  );

  if (resource) {
    const response = await fetch(resource[1]);
    if (response.status === 200) {
      return await response.arrayBuffer();
    }
  }

  throw new Error(`Failed to load font data for ${font} weight ${weight}`);
}

// Image metadata
export const size = {
  width: 1200,
  height: 630,
};

export const contentType = "image/png";

// Image generation
export default async function Image({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  const t = await getTranslations({ locale });

  // Get translated content
  const name = t("name.full");
  const subtitle = t("subtitle");
  const headline = t("headline");
  const welcome = t("welcome");
  const ogImageUrl = t("ogImageUrl");

  // Split headline into two lines
  const headlineLines = headline.split("\n");

  // Collect all text that will be rendered for font subsetting
  const allText = `${ogImageUrl}${welcome}${name}${subtitle}${headlineLines.join("")}`;

  // Load Inter font from Google Fonts dynamically
  // Using the same font as the rest of the app (via next/font/google)
  const [interMedium, interBold, interBlack] = await Promise.all([
    loadGoogleFont("Inter", 500, allText),
    loadGoogleFont("Inter", 700, allText),
    loadGoogleFont("Inter", 900, allText),
  ]);

  // Load avatar image from public folder
  const avatarPath = join(process.cwd(), "public", siteConfig.avatarUrl);
  const avatarData = await readFile(avatarPath);
  const avatarSrc = `data:image/png;base64,${avatarData.toString("base64")}`;

  return new ImageResponse(
    (
      <div
        style={{
          height: "100%",
          width: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          backgroundColor: "#1a1a1a",
          backgroundImage: `
            linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px)
          `,
          backgroundSize: "20px 20px",
          fontFamily: "Inter",
          padding: "80px",
        }}
      >
        {/* Top URL bubble */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            marginBottom: "16px",
          }}
        >
          <div
            style={{
              border: "2px solid #ffffff",
              borderRadius: "9999px",
              padding: "6px 12px",
              fontSize: "16px",
              color: "#ffffff",
              fontWeight: 500,
              backgroundColor: "transparent",
              lineHeight: "1.2",
              letterSpacing: "0.02em",
            }}
          >
            {ogImageUrl}
          </div>
        </div>

        {/* Welcome text */}
        <div
          style={{
            fontSize: "28px",
            fontWeight: 700,
            color: "#ffffff",
            marginBottom: "40px",
            textAlign: "center",
            lineHeight: "1.3",
          }}
        >
          {welcome}
        </div>

        {/* Main content block */}
        <div
          style={{
            display: "flex",
            flexDirection: "row",
            alignItems: "center",
            backgroundColor: "#000000",
            borderRadius: "24px",
            padding: "60px 80px",
            width: "100%",
            maxWidth: "1200px",
            gap: "60px",
          }}
        >
          {/* Left side - Text content */}
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              flex: 1,
              gap: "12px",
            }}
          >
            {/* Name */}
            <div
              style={{
                fontSize: "64px",
                fontWeight: 700,
                color: "#ffffff",
                lineHeight: "1.15",
                letterSpacing: "-0.01em",
              }}
            >
              {name}
            </div>

            {/* Subtitle */}
            <div
              style={{
                fontSize: "28px",
                color: "#ffffff",
                fontWeight: 400,
                opacity: 0.75,
              }}
            >
              {subtitle}
            </div>

            {/* Headline lines */}
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "2px",
              }}
            >
              {headlineLines.map((line, index) => (
                <div
                  key={index}
                  style={{
                    fontSize: "24px",
                    color: "#ffffff",
                    fontWeight: 400,
                    lineHeight: "1.5",
                  }}
                >
                  {line}
                </div>
              ))}
            </div>
          </div>

          {/* Right side - Avatar */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <img
              src={avatarSrc}
              alt="Avatar"
              width={200}
              height={200}
              style={{
                borderRadius: "50%",
                objectFit: "cover",
              }}
            />
          </div>
        </div>
      </div>
    ),
    {
      ...size,
      fonts: [
        {
          name: "Inter",
          data: interMedium,
          style: "normal",
          weight: 500,
        },
        {
          name: "Inter",
          data: interBold,
          style: "normal",
          weight: 700,
        },
        {
          name: "Inter",
          data: interBlack,
          style: "normal",
          weight: 900,
        },
      ],
    },
  );
}

// Alt text - will be dynamically generated based on locale
// This is a fallback that Next.js will use
export const alt = "Portfolio";
