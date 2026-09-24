"use client";

import { useEffect, useState } from "react";

import { Icons } from "@/components/icons";

interface ViewCountProps {
  path: string;
}

interface CachedViewData {
  views: number;
  timestamp: number;
}

const CACHE_DURATION = 24 * 60 * 60 * 1000; // 24 hours in milliseconds

function getCacheKey(path: string): string {
  return `view-count-${path}`;
}

function getCachedViews(path: string): CachedViewData | null {
  if (typeof window === "undefined") {
    return null;
  }

  try {
    const cacheKey = getCacheKey(path);
    const cached = localStorage.getItem(cacheKey);
    if (!cached) {
      return null;
    }

    const data: CachedViewData = JSON.parse(cached);
    const now = Date.now();
    const age = now - data.timestamp;

    // Check if cache is still valid (within 24 hours)
    if (age < CACHE_DURATION) {
      return data;
    }

    // Cache expired, remove it
    localStorage.removeItem(cacheKey);
    return null;
  } catch (error) {
    console.error("Failed to read view count cache:", error);
    return null;
  }
}

function setCachedViews(path: string, views: number): void {
  if (typeof window === "undefined") {
    return;
  }

  try {
    const cacheKey = getCacheKey(path);
    const data: CachedViewData = {
      views,
      timestamp: Date.now(),
    };
    localStorage.setItem(cacheKey, JSON.stringify(data));
  } catch (error) {
    console.error("Failed to cache view count:", error);
  }
}

export function ViewCount({ path }: ViewCountProps) {
  const [views, setViews] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check cache first
    const cached = getCachedViews(path);
    if (cached) {
      setViews(cached.views);
      setIsLoading(false);
      return;
    }

    // Cache miss or expired, fetch from API
    async function fetchViews() {
      try {
        const response = await fetch(
          `/api/analytics?path=${encodeURIComponent(path)}`,
        );
        if (response.ok) {
          const data = await response.json();
          const viewCount = data.views || 0;
          setViews(viewCount);
          // Cache the result
          setCachedViews(path, viewCount);
        } else {
          setViews(0);
        }
      } catch (error) {
        console.error("Failed to fetch view count:", error);
        setViews(0);
      } finally {
        setIsLoading(false);
      }
    }

    fetchViews();
  }, [path]);

  // Show placeholder during loading to prevent layout shift
  const displayViews = isLoading ? "..." : views?.toLocaleString() || "0";

  return (
    <span className="inline-flex items-center gap-1 text-sm text-neutral-600 dark:text-neutral-400">
      <Icons.eye className="h-4 w-4" />
      <span>{displayViews}</span>
    </span>
  );
}
