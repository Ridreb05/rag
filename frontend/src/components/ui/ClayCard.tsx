import type { HTMLAttributes } from "react";

interface ClayCardProps extends HTMLAttributes<HTMLDivElement> {
  hover?: boolean;
  glass?: boolean;
  radius?: "clay" | "lg" | "xl";
}

const radiusClasses: Record<NonNullable<ClayCardProps["radius"]>, string> = {
  clay: "rounded-[32px]",
  lg: "rounded-[24px]",
  xl: "rounded-[48px]",
};

export function ClayCard({ hover = false, glass = true, radius = "clay", className = "", children, ...props }: ClayCardProps) {
  return (
    <div
      className={`relative overflow-hidden ${radiusClasses[radius]} ${glass ? "bg-white/70 backdrop-blur-xl" : "bg-white"} p-8 text-clay-foreground shadow-clayCard transition-all duration-500 ${
        hover ? "hover:-translate-y-2 hover:shadow-clayCardHover" : ""
      } ${className}`}
      {...props}
    >
      <div className="relative z-10 flex h-full flex-col">{children}</div>
    </div>
  );
}
