import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: {
          base: "#1C1C1E",
          elevated: "#2C2C2E",
          grouped: "#3A3A3C",
          tertiary: "#48484A",
        },
        text: {
          primary: "#FFFFFF",
          secondary: "#8E8E93",
          tertiary: "#636366",
        },
        accent: {
          DEFAULT: "#6C5CE7",
          hover: "#7C6CF7",
          muted: "rgba(108, 92, 231, 0.15)",
        },
        success: "#30D158",
        warning: "#FFD60A",
        error: "#FF453A",
        info: "#64D2FF",
      },
      fontFamily: {
        system: [
          "-apple-system",
          "BlinkMacSystemFont",
          "SF Pro Display",
          "system-ui",
          "sans-serif",
        ],
        mono: ["ui-monospace", "SF Mono", "Cascadia Code", "monospace"],
      },
      borderRadius: {
        apple: "12px",
      },
    },
  },
  plugins: [],
};

export default config;
