"use client";

import { useEffect } from "react";

import { usePathname } from "@/i18n/routing";

/**
 * Component to restore scroll position after language switch
 * Reads scroll position from sessionStorage and restores it after page load
 */
export function ScrollRestore() {
  const pathname = usePathname();

  useEffect(() => {
    // Restore scroll position after page is rendered
    const restoreScroll = () => {
      if (typeof window !== "undefined") {
        const scrollKey = `scroll-${pathname}`;
        const savedScrollPosition = sessionStorage.getItem(scrollKey);

        if (savedScrollPosition !== null) {
          const scrollPosition = parseInt(savedScrollPosition, 10);
          
          // Use multiple requestAnimationFrame calls to ensure DOM is fully rendered
          requestAnimationFrame(() => {
            requestAnimationFrame(() => {
              window.scrollTo({
                top: scrollPosition,
                behavior: "instant", // Instant scroll to avoid animation
              });
              
              // Clear the saved position after restoring
              sessionStorage.removeItem(scrollKey);
            });
          });
        }
      }
    };

    // Wait for next tick to ensure React hydration is complete
    const timeoutId = setTimeout(restoreScroll, 100);

    return () => clearTimeout(timeoutId);
  }, [pathname]);

  return null;
}

