module.exports = {
  version: "2.0",
  title: "Glitchframe",
  description: "Local GPU music video generator (Gradio UI)",
  icon: "icon.png",
  pre: [
    {
      text: "ffmpeg",
      description:
        "Required for video encode/mux. Install so ffmpeg is on your PATH (e.g. winget on Windows).",
      href: "https://ffmpeg.org/download.html",
    },
  ],
  menu: async (kernel, info) => {
    const installing = info.running("install.js");
    const installed = info.exists("env");

    if (installing) {
      return [
        { icon: "fa-solid fa-plug", text: "Installing...", href: "install.js" },
      ];
    }

    if (installed) {
      const running = info.running("start.js");
      if (running) {
        const memory = info.local("start.js");
        if (memory && memory.url) {
          return [
            { icon: "fa-solid fa-rocket", text: "Open Web UI", href: memory.url },
            { icon: "fa-solid fa-terminal", text: "Terminal", href: "start.js" },
            { icon: "fa-solid fa-rotate", text: "Update", href: "update.js" },
            { icon: "fa-solid fa-plug", text: "Reinstall", href: "install.js" },
            { icon: "fa-solid fa-broom", text: "Factory Reset", href: "reset.js" },
          ];
        }
        return [
          { icon: "fa-solid fa-terminal", text: "Terminal", href: "start.js" },
          { icon: "fa-solid fa-rotate", text: "Update", href: "update.js" },
          { icon: "fa-solid fa-broom", text: "Factory Reset", href: "reset.js" },
        ];
      }
      return [
        { icon: "fa-solid fa-power-off", text: "Start", href: "start.js" },
        { icon: "fa-solid fa-rotate", text: "Update", href: "update.js" },
        { icon: "fa-solid fa-plug", text: "Reinstall", href: "install.js" },
        { icon: "fa-solid fa-broom", text: "Factory Reset", href: "reset.js" },
      ];
    }

    return [
      { icon: "fa-solid fa-plug", text: "Install", href: "install.js" },
      { icon: "fa-solid fa-rotate", text: "Update", href: "update.js" },
    ];
  },
};
