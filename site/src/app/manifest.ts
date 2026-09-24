import type { MetadataRoute } from "next";

import enMessages from "@/i18n/messages/en/personal.json";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: enMessages.name.full,
    short_name: enMessages.name.full,
    description: enMessages.headline.replace(/\n/g, ", "),
    start_url: "/",
    display: "standalone",
    background_color: "#ffffff",
    theme_color: "#000000",
    orientation: "portrait-primary",
    scope: "/",
    lang: "en",
    icons: [
      {
        src: "/favicon.ico",
        sizes: "any",
        type: "image/x-icon",
      },
    ],
  };
}
