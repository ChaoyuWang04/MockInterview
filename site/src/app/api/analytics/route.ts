// app/api/analytics/route.ts
// Fetch total GA4 sessions securely via service account
// Also supports fetching page views for specific paths

import { BetaAnalyticsDataClient } from "@google-analytics/data";
import { GoogleAuth } from "google-auth-library";
import { NextRequest, NextResponse } from "next/server";

// Enable Next.js incremental static regeneration for this route
// This makes the response cached for 24 hours and regenerated in the background
export const revalidate = 86400; // 86400 seconds = 24 hours

const propertyId = process.env.GA4_PROPERTY_ID!;
const clientEmail = process.env.GA4_CLIENT_EMAIL!;
const privateKey = (process.env.GA4_PRIVATE_KEY || "").replace(/\\n/g, "\n");

export async function GET(request: NextRequest) {
  try {
    // Check environment variables
    if (!propertyId || !clientEmail || !privateKey) {
      return NextResponse.json(
        { error: "Missing GA4 credentials" },
        { status: 500 },
      );
    }

    // Get path parameter from query string
    const searchParams = request.nextUrl.searchParams;
    const path = searchParams.get("path");

    // Initialize GA4 Data API client
    // Using GoogleAuth with service account credentials
    // Reference: https://docs.cloud.google.com/docs/authentication/client-libraries#node.js
    const auth = new GoogleAuth({
      credentials: {
        client_email: clientEmail,
        private_key: privateKey,
      },
      scopes: ["https://www.googleapis.com/auth/analytics.readonly"],
    });

    const analyticsDataClient = new BetaAnalyticsDataClient({
      // Type assertion needed due to version mismatch between google-auth-library versions
      // @google-analytics/data uses 10.4.0, but we have 10.5.0 installed
      // Runtime compatibility is maintained, only TypeScript types differ
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      auth: auth as any,
    });

    // If path is provided, query for specific page views
    if (path) {
      // Normalize path: ensure it starts with /
      const normalizedPath = path.startsWith("/") ? path : `/${path}`;
      
      // Query GA4 for page views of specific path
      const pageViewsResponse = await analyticsDataClient.runReport({
        property: `properties/${propertyId}`,
        dateRanges: [{ startDate: "2025-01-01", endDate: "today" }],
        dimensions: [{ name: "pagePath" }],
        metrics: [{ name: "screenPageViews" }],
        dimensionFilter: {
          filter: {
            fieldName: "pagePath",
            stringFilter: {
              matchType: "EXACT",
              value: normalizedPath,
            },
          },
        },
      });

      const pageViews = pageViewsResponse[0]?.rows?.[0]?.metricValues?.[0]
        ?.value
        ? Number(pageViewsResponse[0].rows[0].metricValues[0].value)
        : 0;

      // Add caching headers (edge + browser) - cache for 24 hours
      const headers = new Headers({
        "Cache-Control": "public, s-maxage=86400, stale-while-revalidate=3600",
      });

      return NextResponse.json(
        {
          path: normalizedPath,
          views: pageViews,
        },
        { headers },
      );
    }

    // Default: Query GA4 for total sessions only
    const sessionsResponse = await analyticsDataClient.runReport({
      property: `properties/${propertyId}`,
      dateRanges: [{ startDate: "2025-01-01", endDate: "today" }],
      metrics: [{ name: "sessions" }],
    });

    const totalSessions = sessionsResponse[0]?.rows?.[0]?.metricValues?.[0]
      ?.value
      ? Number(sessionsResponse[0].rows[0].metricValues[0].value)
      : 0;

    // Add caching headers (edge + browser) - cache for 24 hours
    const headers = new Headers({
      "Cache-Control": "public, s-maxage=86400, stale-while-revalidate=3600",
    });

    return NextResponse.json(
      {
        metric: "sessions",
        label: "Total Sessions",
        total: totalSessions,
      },
      { headers },
    );
  } catch (err: unknown) {
    console.error("GA4 API Error:", err);
    return NextResponse.json(
      {
        error: "Failed to fetch GA4 data",
        detail: String(err),
      },
      { status: 500 },
    );
  }
}
