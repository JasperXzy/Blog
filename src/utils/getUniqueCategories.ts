import type { CollectionEntry } from "astro:content";
import { slugifyStr } from "./slugify";

export type Category = {
  category: string;
  categoryName: string;
};

export const getUniqueCategories = (
  posts: CollectionEntry<"posts">[]
): Category[] =>
  posts
    .map(post => ({
      category: slugifyStr(post.data.category),
      categoryName: post.data.category,
    }))
    .filter(
      (value, index, self) =>
        self.findIndex(item => item.category === value.category) === index
    )
    .sort((a, b) => a.category.localeCompare(b.category));
