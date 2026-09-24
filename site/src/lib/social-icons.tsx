import { Icons } from "@/components/icons";

export type IconKey = keyof typeof Icons;

/**
 * Maps icon key strings to icon components
 */
export function getSocialIcon(
  iconKey: string,
): React.ComponentType<{ className?: string }> {
  const icon = Icons[iconKey as IconKey];
  if (!icon) {
    console.warn(`Icon "${iconKey}" not found, falling back to default`);
    return Icons.globe;
  }
  return icon;
}

/**
 * Converts social data from i18n (with icon keys as strings) to format expected by components
 */
export function transformSocialData(
  socialData: Record<
    string,
    {
      name: string;
      url: string;
      icon: string;
      navbar?: boolean;
      content?: boolean;
      footer?: boolean;
    }
  >,
): Record<
  string,
  {
    name: string;
    url: string;
    icon: React.ComponentType<{ className?: string }>;
    navbar?: boolean;
    content?: boolean;
    footer?: boolean;
  }
> {
  const transformed: Record<
    string,
    {
      name: string;
      url: string;
      icon: React.ComponentType<{ className?: string }>;
      navbar?: boolean;
      content?: boolean;
      footer?: boolean;
    }
  > = {};

  for (const [key, value] of Object.entries(socialData)) {
    transformed[key] = {
      ...value,
      icon: getSocialIcon(value.icon),
    };
  }

  return transformed;
}
