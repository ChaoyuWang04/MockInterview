import { getTranslations } from "next-intl/server";
import type {
  BlogPosting,
  BreadcrumbList,
  Country,
  EducationalOrganization,
  Occupation,
  Organization,
  Person,
  PostalAddress,
  WebSite,
  WithContext,
} from "schema-dts";

import { siteConfig } from "@/data/site";
import { DEFAULT_LOCALE, getLocaleUrl, type Locale } from "@/i18n/routing";

/*
 * Generate BreadcrumbList JSON-LD
 */

export async function generateBreadcrumbListJsonLd(
  locale: Locale = DEFAULT_LOCALE,
): Promise<string> {
  const t = await getTranslations({ locale });

  const navbarItems = t.raw("navbar.items") as Array<{
    href: string;
    icon: string;
    label: string;
  }>;

  const breadcrumbList: WithContext<BreadcrumbList> = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: navbarItems.map((item, index) => ({
      "@type": "ListItem",
      position: index + 1,
      name: item.label,
      item: getLocaleUrl(locale, item.href),
    })),
  };

  return JSON.stringify(breadcrumbList);
}

/*
 * Generate Website JSON-LD
 */

export async function generateWebsiteJsonLd(
  locale: Locale = DEFAULT_LOCALE,
): Promise<string> {
  const t = await getTranslations({ locale });
  const baseUrl = getLocaleUrl(locale);
  const author: Person = {
    "@type": "Person",
    name: t("name.full"),
    url: baseUrl,
  };

  const headline = t("headline").replace(/\n/g, ", ");
  const websiteJsonLd: WithContext<WebSite> = {
    "@context": "https://schema.org",
    "@type": "WebSite",
    name: t("name.full"),
    description: headline,
    url: baseUrl,
    author,
  };

  return JSON.stringify(websiteJsonLd);
}

/*
 * Generate Person JSON-LD
 */

async function getSocialMediaUrls(
  locale: Locale = DEFAULT_LOCALE,
): Promise<string[]> {
  const t = await getTranslations({ locale });
  const socialData = t.raw("social") as Record<
    string,
    {
      name: string;
      url: string;
      icon: string;
      navbar?: boolean;
      content?: boolean;
      footer?: boolean;
    }
  >;
  return Object.values(socialData)
    .filter((social) => social.url && social.name !== "Email")
    .map((social) => social.url);
}

async function getAddress(
  locale: Locale = DEFAULT_LOCALE,
): Promise<PostalAddress> {
  const t = await getTranslations({ locale });
  const addressCountry: Country = { "@type": "Country", name: "Singapore" };
  const address: PostalAddress = {
    "@type": "PostalAddress",
    addressLocality: t("location.name"),
    addressCountry,
  };
  return address;
}

async function getOrganization(
  locale: Locale = DEFAULT_LOCALE,
): Promise<Organization | undefined> {
  const t = await getTranslations({ locale });
  try {
    const workItems = t.raw("work.items") as Array<{
      company: string;
      href: string;
      title: string;
      description: string;
    }>;
    if (!workItems || workItems.length === 0) {
      return undefined;
    }
    const organization: Organization = {
      "@type": "Organization",
      name: workItems[0].company,
      url: workItems[0].href,
    };
    return organization;
  } catch {
    return undefined;
  }
}

async function getOccupation(
  locale: Locale = DEFAULT_LOCALE,
): Promise<Occupation | undefined> {
  const t = await getTranslations({ locale });
  try {
    const workItems = t.raw("work.items") as Array<{
      company: string;
      href: string;
      title: string;
      description: string;
    }>;
    if (!workItems || workItems.length === 0) {
      return undefined;
    }
    return {
      "@type": "Occupation",
      name: workItems[0].title,
      description: workItems[0].description,
    };
  } catch {
    return undefined;
  }
}

async function getEducation(
  locale: Locale = DEFAULT_LOCALE,
): Promise<EducationalOrganization[]> {
  const t = await getTranslations({ locale });
  try {
    const educationItems = t.raw("education.items") as Array<{
      school: string;
      href: string;
      degree: string;
      logoUrl: string;
      start: string;
      end: string;
    }>;
    if (!educationItems || !Array.isArray(educationItems)) {
      return [];
    }
    return educationItems.map((edu) => ({
      "@type": "EducationalOrganization",
      name: edu.school,
      url: edu.href,
    }));
  } catch {
    return [];
  }
}

export async function generatePersonJsonLd(
  locale: Locale = DEFAULT_LOCALE,
): Promise<string> {
  const t = await getTranslations({ locale });
  const headline = t("headline").replace(/\n/g, ", ");
  const baseUrl = getLocaleUrl(locale);
  const socialData = t.raw("social") as Record<
    string,
    {
      name: string;
      url: string;
      icon: string;
      navbar?: boolean;
      content?: boolean;
      footer?: boolean;
    }
  >;
  // Safely get skills array
  let skills: string[] = [];
  try {
    const skillsValue = t.raw("skills");
    if (Array.isArray(skillsValue)) {
      skills = skillsValue as string[];
    }
  } catch {
    // If skills doesn't exist, use empty array
  }
  
  // Safely get job title from work items
  let jobTitle: string | undefined;
  try {
    const workItems = t.raw("work.items") as Array<{ title: string }>;
    if (workItems && workItems.length > 0) {
      jobTitle = workItems[0].title;
    }
  } catch {
    // If work.items doesn't exist, jobTitle remains undefined
  }

  const personJsonLd: Record<string, unknown> = {
    "@context": "https://schema.org",
    "@type": "Person",
    name: t("name.full"),
    givenName: t("name.given"),
    familyName: t("name.family"),
    alternateName: t("name.full"),
    description: headline,
    url: baseUrl,
    image: `${siteConfig.url}${siteConfig.avatarUrl}`,
    /* Contact Info */
    email: socialData.email.url,
    sameAs: await getSocialMediaUrls(locale),
    address: await getAddress(locale),
    /* Education Info */
    alumniOf: await getEducation(locale),
  };

  // Only add knowsAbout if skills exist
  if (skills.length > 0) {
    personJsonLd.knowsAbout = skills;
  }

  // Only add work-related fields if work items exist
  if (jobTitle) {
    personJsonLd.jobTitle = jobTitle;
  }
  const organization = await getOrganization(locale);
  if (organization) {
    personJsonLd.worksFor = organization;
  }
  const occupation = await getOccupation(locale);
  if (occupation) {
    personJsonLd.hasOccupation = occupation;
  }

  return JSON.stringify(personJsonLd);
}

/*
 * Generate Blog JSON-LD
 */
export function generateBlogJsonLd(
  posts: {
    metadata: Record<string, unknown> & {
      title: string;
      date: string;
      summary: string;
    };
    slug: string;
    locale?: Locale;
  }[],
): string {
  const itemListElements = posts
    .filter((post) => !post.locale || post.locale === DEFAULT_LOCALE)
    .map((post, index) => {
      const locale = (post.locale || DEFAULT_LOCALE) as Locale;
      const postUrl = getLocaleUrl(locale, `/blog/${post.slug}`);

      return {
        "@type": "ListItem",
        position: index + 1,
        url: postUrl,
      };
    });

  return JSON.stringify({
    "@context": "https://schema.org",
    "@type": "ItemList",
    itemListElement: itemListElements,
  });
}

/*
 * Generate BlogPosting JSON-LD
 */

export async function generateBlogPostingJsonLd(post: {
  metadata: Record<string, unknown> & {
    title: string;
    date: string;
    summary: string;
    image?: string;
  };
  slug: string;
  locale?: Locale;
}): Promise<string> {
  const locale = (post.locale || DEFAULT_LOCALE) as Locale;
  const socialMediaUrls = await getSocialMediaUrls(locale);
  const t = await getTranslations({ locale });
  const baseUrl = getLocaleUrl(locale);
  const postUrl = getLocaleUrl(locale, `/blog/${post.slug}`);

  const author: Person = {
    "@type": "Person",
    name: t("name.full"),
    url: baseUrl,
    sameAs: socialMediaUrls,
  };

  const publisher: Person = {
    "@type": "Person",
    name: t("name.full"),
    url: baseUrl,
  };

  const blogPosting: WithContext<BlogPosting> = {
    "@context": "https://schema.org",
    "@type": "BlogPosting",
    headline: post.metadata.title,
    datePublished: post.metadata.date,
    dateModified: post.metadata.date,
    description: post.metadata.summary,
    image: post.metadata.image
      ? `${siteConfig.url}${post.metadata.image}`
      : `${postUrl}/opengraph-image`,
    url: postUrl,
    author,
    publisher,
    mainEntityOfPage: {
      "@type": "WebPage",
      "@id": postUrl,
    },
  };

  return JSON.stringify(blogPosting);
}
