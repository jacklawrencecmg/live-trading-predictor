/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        surface: "#0f1117",
        panel: "#161b22",
        border: "#21262d",
        accent: "#58a6ff",
        green: { trade: "#3fb950" },
        red: { trade: "#f85149" },
      },
    },
  },
  plugins: [],
};
