import { defineAstroPaperConfig } from "./src/types/config";

export default defineAstroPaperConfig({
  site: {
    url: "https://blog.jasperxzy.com/",
    title: "Zhengyi's Blog",
    description: "记录 AI、SLAM、机器人与嵌入式 AI 的学习、研究和工程实践。",
    author: "Zhengyi Xu",
    profile: "https://jasperxzy.com",
    ogImage: "default-og.jpg",
    lang: "zh-CN",
    timezone: "Asia/Shanghai",
    dir: "ltr",
  },
  posts: {
    perPage: 4,
    perIndex: 4,
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
  shareLinks: [
    { name: "x",        url: "https://x.com/intent/post?url=" },
    { name: "telegram", url: "https://t.me/share/url?url=" },
    { name: "mail",     url: "mailto:?subject=See%20this%20post&body=" },
  ],
});
