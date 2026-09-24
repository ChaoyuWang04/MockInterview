import Link from "next/link";

import { cn } from "@/lib/utils";

interface ContactProps {
  emailUrl: string;
  calendlyUrl?: string;
  contactLabel?: string;
  getInTouch?: string;
  contactDescription?: string;
  via?: string;
  emailLabel?: string;
  or?: string;
  zoomLabel?: string;
  askQuestions?: string;
  exploreCollaboration?: string;
  className?: string;
  align?: "left" | "center";
}

export default function Contact({
  emailUrl,
  calendlyUrl,
  contactLabel = "Contact",
  getInTouch = "Get in Touch",
  contactDescription = "Want to chat? Feel free to reach out",
  via = "via",
  emailLabel = "email",
  or = "or",
  zoomLabel = "Zoom",
  askQuestions = "Ask questions",
  exploreCollaboration = "Explore collaboration opportunities",
  className,
  align = "center",
}: ContactProps) {
  const isLeftAligned = align === "left";

  return (
    <div className={cn("space-y-3", className)}>
      <div className="bg-foreground text-background inline-block rounded-lg px-3 py-1 text-sm">
        {contactLabel}
      </div>
      <h2 className="text-3xl font-bold tracking-tighter sm:text-5xl">
        {getInTouch}
      </h2>
      <div
        className={cn(
          "space-y-6",
          isLeftAligned ? "max-w-[34rem]" : "mx-auto max-w-[600px]",
        )}
      >
        <p
          className={cn(
            "text-muted-foreground text-lg leading-relaxed md:text-xl",
            isLeftAligned ? "text-left" : "text-center",
          )}
        >
          {contactDescription}{" "}
          {via}{" "}
          <Link
            href={emailUrl}
            className="inline-flex items-center gap-1 text-foreground underline transition-colors hover:no-underline"
          >
            {emailLabel}
          </Link>
          {calendlyUrl ? (
            <>
              {" "}
              {or}{" "}
              <Link
                href={calendlyUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 text-foreground underline transition-colors hover:no-underline"
              >
                {zoomLabel}
              </Link>{" "}
              <span className="inline-block">→</span>
            </>
          ) : (
            <>
              {" "}
              <span className="inline-block">→</span>
            </>
          )}
        </p>

        <div
          className={cn(
            "flex flex-col space-y-4",
            isLeftAligned ? "items-start" : "items-center",
          )}
        >
          <ul
            className={cn(
              "text-muted-foreground grid gap-3 text-lg leading-relaxed md:text-xl",
              isLeftAligned ? "text-left" : "text-center",
            )}
          >
            <li className="hover:text-foreground transition-colors">
              • {askQuestions}
            </li>
            <li className="hover:text-foreground transition-colors">
              • {exploreCollaboration}
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
