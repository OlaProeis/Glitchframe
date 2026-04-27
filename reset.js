module.exports = {
  run: [
    {
      method: "fs.rm",
      params: {
        path: "env",
        recursive: true,
      },
    },
    {
      method: "notify",
      params: {
        html: "Removed the <code>env</code> folder. Run <b>Install</b> again from the sidebar.",
      },
    },
  ],
};
