"use client";

import { Play } from "lucide-react";
import Image from "next/image";
import { useEffect, useMemo, useRef, useState } from "react";

import {
  type ContactMediaItem,
  getYouTubeEmbedUrl,
  getYouTubePosterUrl,
} from "@/data/contact-media";
import { cn } from "@/lib/utils";

interface ContactMediaWallProps {
  items: readonly ContactMediaItem[];
  playVideoLabel: string;
  ariaLabel: string;
  className?: string;
}

function ContactMediaCard({
  item,
  playVideoLabel,
}: {
  item: ContactMediaItem;
  playVideoLabel: string;
}) {
  const [isPlaying, setIsPlaying] = useState(false);

  if (item.type === "image") {
    return (
      <div className="bg-muted relative aspect-video overflow-hidden rounded-2xl">
        <Image
          src={item.src}
          alt={item.alt}
          fill
          sizes="(max-width: 1024px) 100vw, 33vw"
          className="object-cover"
        />
      </div>
    );
  }

  if (isPlaying) {
    return (
      <div className="bg-muted relative aspect-video overflow-hidden rounded-2xl">
        <iframe
          src={getYouTubeEmbedUrl(item.youtubeId)}
          title={item.title}
          className="absolute inset-0 h-full w-full"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
          referrerPolicy="strict-origin-when-cross-origin"
          allowFullScreen
        />
      </div>
    );
  }

  const posterUrl = item.poster || getYouTubePosterUrl(item.youtubeId);

  return (
    <button
      type="button"
      onClick={() => setIsPlaying(true)}
      className="bg-muted group relative block aspect-video w-full overflow-hidden rounded-2xl text-left"
      aria-label={`${playVideoLabel}: ${item.title}`}
    >
      <div
        className="absolute inset-0 bg-cover bg-center transition-transform duration-500 group-hover:scale-[1.02]"
        style={{ backgroundImage: `url(${posterUrl})` }}
        aria-hidden
      />
      <div
        className="absolute inset-0 bg-gradient-to-t from-black/45 via-black/10 to-transparent"
        aria-hidden
      />
      <div className="absolute inset-x-0 bottom-0 flex items-end justify-between p-4 text-white">
        <span className="line-clamp-2 text-sm font-medium">{item.title}</span>
        <span className="inline-flex size-11 shrink-0 items-center justify-center rounded-full border border-white/15 bg-black/70">
          <Play className="ml-0.5 size-4 fill-current" />
        </span>
      </div>
    </button>
  );
}

export default function ContactMediaWall({
  items,
  playVideoLabel,
  ariaLabel,
  className,
}: ContactMediaWallProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isPaused, setIsPaused] = useState(false);
  const duplicatedItems = useMemo(() => [...items, ...items], [items]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || items.length === 0) {
      return;
    }

    let animationFrameId = 0;
    let previousTimestamp = 0;
    const speed = 0.04;

    const tick = (timestamp: number) => {
      if (previousTimestamp === 0) {
        previousTimestamp = timestamp;
      }

      const delta = timestamp - previousTimestamp;
      previousTimestamp = timestamp;

      if (!isPaused) {
        container.scrollTop += delta * speed;
        const loopHeight = container.scrollHeight / 2;

        if (container.scrollTop >= loopHeight) {
          container.scrollTop -= loopHeight;
        }
      }

      animationFrameId = window.requestAnimationFrame(tick);
    };

    animationFrameId = window.requestAnimationFrame(tick);

    return () => {
      window.cancelAnimationFrame(animationFrameId);
    };
  }, [isPaused, items.length]);

  return (
    <div
      ref={containerRef}
      className={cn(
        "scrollbar-thin h-[15rem] overflow-y-auto overflow-x-hidden rounded-[1.5rem] pr-1 sm:h-[20rem] lg:h-[22rem]",
        className,
      )}
      aria-label={ariaLabel}
      onMouseEnter={() => setIsPaused(true)}
      onMouseLeave={() => setIsPaused(false)}
      onTouchStart={() => setIsPaused(true)}
      onTouchEnd={() => setIsPaused(false)}
    >
      <div className="grid gap-4">
        {duplicatedItems.map((item, index) => (
          <ContactMediaCard
            key={`${item.type}-${index}`}
            item={item}
            playVideoLabel={playVideoLabel}
          />
        ))}
      </div>
    </div>
  );
}
