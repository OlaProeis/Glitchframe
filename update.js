module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: ["git pull"],
      },
    },
    {
      method: "notify",
      params: {
        html: "Repository updated. If dependencies changed, run <b>Install</b> again or use the in-app terminal.",
      },
    },
  ],
};
