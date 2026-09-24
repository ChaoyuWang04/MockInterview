import Link from "next/link";

interface Social {
  name: string;
  url: string;
  icon: React.ComponentType<{ className?: string }>;
  navbar?: boolean;
  content?: boolean;
  footer?: boolean;
}

export default function SocialLinks({
  socials,
  className = "text-muted-foreground hover:text-foreground",
}: {
  socials: Record<string, Social>;
  delay?: number;
  className?: string;
}) {
  return (
    <div className="flex items-center justify-center gap-4">
      {Object.values(socials)
        .filter((social) => social.content)
        .map((social) => (
          <Link
            key={social.name}
            href={social.url}
            target="_blank"
            rel="noopener noreferrer"
            className={className}
          >
            <social.icon className="size-5" />
            <span className="sr-only">{social.name}</span>
          </Link>
        ))}
    </div>
  );
}
