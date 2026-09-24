import { Locale } from "@/i18n/routing";
import {
  generateBreadcrumbListJsonLd,
  generateWebsiteJsonLd,
} from "@/lib/jsonld";
import { jsonldScript } from "@/lib/utils";

export default async function JsonLdScripts({ locale }: { locale: Locale }) {
  const [websiteJsonLd, breadcrumbJsonLd] = await Promise.all([
    generateWebsiteJsonLd(locale),
    generateBreadcrumbListJsonLd(locale),
  ]);

  return (
    <>
      {jsonldScript(websiteJsonLd)}
      {jsonldScript(breadcrumbJsonLd)}
    </>
  );
}
