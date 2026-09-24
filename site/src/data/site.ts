/**
 * Site configuration
 * Unified configuration for the portfolio website
 */

export const BLUR_FADE_DELAY = 0.05;

export const siteConfig = {
  url: "https://chaoyuwang.vercel.app", // 不带尾部斜杠:各处按 `${url}/path` 拼接
  lastUpdated: "2026.03",
  avatarUrl: "/me.jpg",
} as const;
