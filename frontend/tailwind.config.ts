import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "media",
  theme: {
    extend: {
      colors: {
        clay: {
          canvas: "#F4F1FA",
          foreground: "#332F3A",
          muted: "#635F69",
          accent: "#7C3AED",
          accentAlt: "#DB2777",
          tertiary: "#0EA5E9",
          success: "#10B981",
          warning: "#F59E0B",
          pressed: "#EFEBF5",
        },
      },
      fontFamily: {
        heading: ["Nunito", "sans-serif"],
        body: ["DM Sans", "sans-serif"],
        script: [
          "Noto Sans Devanagari",
          "Noto Sans Bengali",
          "Noto Sans Gujarati",
          "Noto Sans Gurmukhi",
          "Noto Sans Kannada",
          "Noto Sans Malayalam",
          "Noto Sans Oriya",
          "Noto Sans Tamil",
          "Noto Sans Telugu",
          "Noto Nastaliq Urdu",
          "DM Sans",
          "sans-serif",
        ],
      },
      boxShadow: {
        clayDeep:
          "30px 30px 60px #cdc6d9, -30px -30px 60px #ffffff, inset 10px 10px 20px rgba(139, 92, 246, 0.05), inset -10px -10px 20px rgba(255, 255, 255, 0.8)",
        clayCard:
          "16px 16px 32px rgba(160, 150, 180, 0.2), -10px -10px 24px rgba(255, 255, 255, 0.9), inset 6px 6px 12px rgba(139, 92, 246, 0.03), inset -6px -6px 12px rgba(255, 255, 255, 1)",
        clayCardHover:
          "22px 22px 44px rgba(160, 150, 180, 0.28), -12px -12px 28px rgba(255, 255, 255, 0.95), inset 6px 6px 12px rgba(139, 92, 246, 0.03), inset -6px -6px 12px rgba(255, 255, 255, 1)",
        clayButton:
          "12px 12px 24px rgba(139, 92, 246, 0.3), -8px -8px 16px rgba(255, 255, 255, 0.4), inset 4px 4px 8px rgba(255, 255, 255, 0.4), inset -4px -4px 8px rgba(0, 0, 0, 0.1)",
        clayButtonHover:
          "16px 16px 30px rgba(139, 92, 246, 0.38), -10px -10px 20px rgba(255, 255, 255, 0.5), inset 4px 4px 8px rgba(255, 255, 255, 0.4), inset -4px -4px 8px rgba(0, 0, 0, 0.1)",
        clayPressed: "inset 10px 10px 20px #d9d4e3, inset -10px -10px 20px #ffffff",
      },
      borderRadius: {
        clay: "32px",
      },
      keyframes: {
        "clay-float": {
          "0%, 100%": { transform: "translateY(0) rotate(0deg)" },
          "50%": { transform: "translateY(-20px) rotate(2deg)" },
        },
        "clay-float-delayed": {
          "0%, 100%": { transform: "translateY(0) rotate(0deg)" },
          "50%": { transform: "translateY(-15px) rotate(-2deg)" },
        },
        "clay-float-slow": {
          "0%, 100%": { transform: "translateY(0) rotate(0deg)" },
          "50%": { transform: "translateY(-30px) rotate(5deg)" },
        },
        "clay-breathe": {
          "0%, 100%": { transform: "scale(1)" },
          "50%": { transform: "scale(1.02)" },
        },
        "clay-mic-pulse": {
          "50%": { boxShadow: "0 0 0 14px rgba(124, 58, 237, 0.16)" },
        },
        "clay-radar": {
          "0%": { opacity: "1", transform: "scale(0.3)" },
          "100%": { opacity: "0", transform: "scale(1.15)" },
        },
      },
      animation: {
        "clay-float": "clay-float 8s ease-in-out infinite",
        "clay-float-delayed": "clay-float-delayed 10s ease-in-out infinite",
        "clay-float-slow": "clay-float-slow 12s ease-in-out infinite",
        "clay-breathe": "clay-breathe 6s ease-in-out infinite",
        "clay-mic-pulse": "clay-mic-pulse 1.2s ease-out infinite",
        "clay-radar": "clay-radar 1.6s ease-out infinite",
      },
    },
  },
  plugins: [],
} satisfies Config;
