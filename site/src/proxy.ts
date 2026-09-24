import { NextRequest, NextResponse } from "next/server";
import createIntlMiddleware from "next-intl/middleware";

import { routing } from "@/i18n/routing";

const intlMiddleware = createIntlMiddleware(routing);

export async function proxy(request: NextRequest): Promise<NextResponse> {
  const intlResponse = intlMiddleware(request);

  return intlResponse;
}

export const config = {
  matcher: [
    "/((?!api|trpc|_next|_vercel|(?:zh/)?blog(?:/|$)|open-source/nextjs-portfolio-blog-research|.*\\..*).*)",
  ],
};
