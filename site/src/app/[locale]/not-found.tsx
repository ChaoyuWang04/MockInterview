import { Button } from "@/components/ui/button";
import { Link } from "@/i18n/routing";

export default function NotFound() {
  return (
    <main className="bg-background flex min-h-screen flex-col items-center justify-center px-4">
      <div className="text-center">
        <h1 className="text-foreground mb-4 text-6xl font-bold">404</h1>
        <h2 className="text-foreground mb-4 text-2xl font-semibold">
          Page Not Found
        </h2>
        <p className="text-muted-foreground mb-8 max-w-md">
          Sorry, we couldn't find the page you're looking for. It might have
          been moved, deleted, or you entered the wrong URL.
        </p>
        <div className="flex flex-col justify-center gap-4 sm:flex-row">
          <Button asChild>
            <Link href="/">Go Home</Link>
          </Button>
          <Button variant="outline" asChild>
            <a href="/blog">View Blog</a>
          </Button>
        </div>
      </div>
    </main>
  );
}
