import Image from "next/image";
import type { Components } from "react-markdown";
import ReactMarkdown from "react-markdown";

interface CustomReactMarkdownProps {
  children: string;
  className?: string;
}

// Custom Image component that uses Next.js Image or native img for shields.io
function CustomImage({ src, alt }: React.ImgHTMLAttributes<HTMLImageElement>) {
  if (!src) return null;

  // Use native img tag for shields.io URLs
  if (typeof src === 'string' && src.includes('img.shields.io')) {
    return (
      <img
        src={src as string}
        alt={alt || ""}
        className="object-contain"
      />
    );
  }

  return (
    <Image
      src={src as string}
      alt={alt || ""}
      width={800}
      height={600}
      sizes="(max-width: 768px) 100vw, 768px"
      className="object-contain"
    />
  );
}

// Custom components for react-markdown
const components: Components = {
  img: CustomImage,
};

export function CustomReactMarkdown({
  children,
  className,
}: CustomReactMarkdownProps) {
  return (
    <div className={className}>
      <ReactMarkdown components={components}>{children}</ReactMarkdown>
    </div>
  );
}
