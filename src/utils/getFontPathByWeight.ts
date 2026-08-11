import type { FontData } from "astro:assets";

export function getFontPathByWeight(
  fonts: FontData[],
  weight: number,
  options?: {
    style?: "normal" | "italic";
    format?: string;
  }
): string | undefined {
  const style = options?.style ?? "normal";
  const format = options?.format ?? "truetype";
  const matchingFonts = fonts.filter(
    font => font.weight === String(weight) && font.style === style
  );

  for (const font of matchingFonts) {
    const src = font.src.find(file => file.format === format);
    if (src) return src.url;
  }

  return matchingFonts[0]?.src[0]?.url;
}
