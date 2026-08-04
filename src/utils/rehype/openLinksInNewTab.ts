type HastNode = {
  type: string;
  tagName?: string;
  properties?: Record<string, unknown>;
  children?: HastNode[];
};

/** Open every non-fragment link rendered from Markdown in a new tab. */
export function rehypeOpenLinksInNewTab() {
  return (tree: HastNode) => {
    function visit(node: HastNode) {
      if (node.type === "element" && node.tagName === "a") {
        const href = node.properties?.href;

        if (typeof href === "string" && href && !href.startsWith("#")) {
          node.properties ??= {};
          node.properties.target = "_blank";
          node.properties.rel = ["noopener", "noreferrer"];
        }
      }

      node.children?.forEach(visit);
    }

    visit(tree);
  };
}
