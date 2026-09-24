export type ContactMediaItem =
  | {
      type: "image";
      src: string;
      alt: string;
    }
  | {
      type: "youtube";
      youtubeId: string;
      title: string;
      poster?: string;
    };

export const contactMediaItems: readonly ContactMediaItem[] = [
  {
    type: "image",
    src: "/launchpad2.jpg",
    alt: "",
  },
  {
    type: "image",
    src: "/nufriends.png",
    alt: "",
  },
  {
    type: "image",
    src: "/nufriends2.jpg",
    alt: "",
  },{
    type: "image",
    src: "/nugraduation.png",
    alt: "",
  },{
    type: "image",
    src: "/self1.jpg",
    alt: "",
  },
  {
    type: "image",
    src: "/ucsd1.jpeg",
    alt: "",
  },
  {
    type: "image",
    src: "/WF.jpg",
    alt: "",
  },
];

export function getYouTubeEmbedUrl(youtubeId: string) {
  return `https://www.youtube.com/embed/${youtubeId}?autoplay=1&rel=0`;
}

export function getYouTubePosterUrl(youtubeId: string) {
  return `https://i.ytimg.com/vi/${youtubeId}/hqdefault.jpg`;
}
