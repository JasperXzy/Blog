import { defineAstroPaperConfig } from "./src/types/config";

export default defineAstroPaperConfig({
  site: {
    url: "https://blog.jasperxzy.com/",
    title: "Zhengyi's Blog",
    description:
      "Notes on AI, SLAM, robotics, multimodal place recognition, and embedded AI engineering.",
    author: "Zhengyi Xu",
    profile: "https://jasperxzy.com",
    ogImage: "default-og.jpg",
    lang: "en",
    timezone: "Asia/Shanghai",
    dir: "ltr",
  },
  posts: {
    perPage: 10,
    perIndex: 10,
    scheduledPostMargin: 15 * 60 * 1000,
  },
  features: {
    lightAndDarkMode: true,
    dynamicOgImage: false,
    showArchives: true,
    showBackButton: true,
    editPost: {
      enabled: true,
      url: "https://github.com/JasperXzy/blog/edit/main/",
    },
    search: "pagefind",
  },
  socials: [
    { name: "github", url: "https://github.com/JasperXzy" },
    {
      name: "linkedin",
      url: "https://www.linkedin.com/in/jasperxzy0409/",
    },
    { name: "mail", url: "mailto:jasper.zhengyi.xu@gmail.com" },
  ],
});
