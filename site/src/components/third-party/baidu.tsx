/**
 * Baidu Site Verification component
 */

export function BaiduSiteVerification() {
  const baiduSiteVerification = process.env.NEXT_PUBLIC_BAIDU_SITE_VERIFICATION;

  if (!baiduSiteVerification) {
    return null;
  }

  return (
    <>
      {/* Baidu Site Verification */}
      <meta
        name="baidu-site-verification"
        content={baiduSiteVerification}
      />
    </>
  );
}
