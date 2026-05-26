/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      boxShadow: {
        glow: "0 0 0 1px rgba(255,255,255,0.08), 0 0 30px rgba(99,102,241,0.25)",
      },
      backgroundImage: {
        "hero-gradient":
          "radial-gradient(1200px 600px at 10% 10%, rgba(99,102,241,0.35) 0%, transparent 55%), radial-gradient(900px 500px at 90% 20%, rgba(168,85,247,0.30) 0%, transparent 50%), linear-gradient(180deg, rgba(2,6,23,1) 0%, rgba(2,6,23,1) 100%)",
      },
    },
  },
  plugins: [],
};

