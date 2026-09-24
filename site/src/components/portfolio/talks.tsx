"use client";

import { ChevronDown } from "lucide-react";
import { useEffect, useState } from "react";

import { ResumeCard } from "@/components/portfolio/resume-card";
import { Button } from "@/components/ui/button";

interface Talk {
  host: string;
  title: string;
  date: string;
  url?: string;
  logoUrl?: string;
}

interface TalksProps {
  talks: readonly Talk[];
  delay?: number;
  showAllText?: string;
}

const DEFAULT_DISPLAY_COUNT = 5;

export default function Talks({ talks, showAllText = "Show All" }: TalksProps) {
  const [showAll, setShowAll] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Sort talks by date (newest first)
  const sortedTalks = [...talks].sort((a, b) => {
    // Convert "YYYY.MM" format to comparable values
    const dateA = a.date.replace(".", "");
    const dateB = b.date.replace(".", "");
    return dateB.localeCompare(dateA);
  });

  // Show latest items by default, or all if showAll is true
  const displayedTalks = showAll
    ? sortedTalks
    : sortedTalks.slice(0, DEFAULT_DISPLAY_COUNT);
  const hasMoreTalks = sortedTalks.length > DEFAULT_DISPLAY_COUNT;

  // Prevent hydration mismatch by not rendering until mounted
  if (!mounted) {
    return (
      <div className="flex min-h-0 flex-col gap-y-3">
        {sortedTalks.slice(0, DEFAULT_DISPLAY_COUNT).map((talk) => (
          <ResumeCard
            key={talk.host + talk.date}
            href={talk.url}
            logoUrl={talk.logoUrl || ""}
            altText={talk.host}
            title={talk.host}
            subtitle={talk.title}
            period={talk.date}
            useMarkdown={true}
          />
        ))}
      </div>
    );
  }

  return (
    <div className="flex min-h-0 flex-col gap-y-3">
      {displayedTalks.map((talk) => (
        <ResumeCard
          key={talk.host + talk.date}
          href={talk.url}
          logoUrl={talk.logoUrl || ""}
          altText={talk.host}
          title={talk.host}
          subtitle={talk.title}
          period={talk.date}
          useMarkdown={true}
        />
      ))}
      {hasMoreTalks && !showAll && (
        <div className="flex justify-center pt-1">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowAll(true)}
            className="flex items-center gap-2"
          >
            <ChevronDown className="h-4 w-4" />
            {showAllText}
          </Button>
        </div>
      )}
    </div>
  );
}
