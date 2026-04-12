/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        surface: "#09090b",
        panel: "#111113",
        "panel-2": "#18181b",
        "panel-3": "#1f1f23",
        border: "#27272a",
        "border-2": "#3f3f46",
        "border-3": "#52525b",
        accent: "#60a5fa",
        positive: "#34d399",
        negative: "#f87171",
        caution: "#fbbf24",
        // backward compat aliases
        green: { trade: "#34d399" },
        red: { trade: "#f87171" },
      },
      fontFamily: {
        mono: ["'SF Mono'", "'Fira Code'", "Menlo", "Consolas", "monospace"],
      },
      fontSize: {
        "2xs": ["10px", { lineHeight: "14px" }],
        "3xs": ["9px", { lineHeight: "12px" }],
      },
      borderRadius: {
        inst: "3px",
      },
    },
  },
  plugins: [],
};
