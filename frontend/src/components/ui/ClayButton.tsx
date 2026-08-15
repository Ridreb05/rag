import { forwardRef } from "react";
import type { ButtonHTMLAttributes } from "react";

type Variant = "primary" | "secondary" | "outline" | "ghost";
type Size = "sm" | "default" | "lg";

interface ClayButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
}

const variantClasses: Record<Variant, string> = {
  primary: "bg-gradient-to-br from-[#A78BFA] to-[#7C3AED] text-white shadow-clayButton hover:shadow-clayButtonHover",
  secondary: "bg-white text-clay-foreground shadow-clayButton hover:shadow-clayButtonHover",
  outline: "border-2 border-clay-accent/20 bg-transparent text-clay-accent hover:border-clay-accent hover:bg-clay-accent/5",
  ghost: "text-clay-foreground hover:bg-clay-accent/10 hover:text-clay-accent",
};

const sizeClasses: Record<Size, string> = {
  sm: "h-11 px-5 text-sm",
  default: "h-14 px-7 text-base",
  lg: "h-16 px-9 text-lg",
};

export const ClayButton = forwardRef<HTMLButtonElement, ClayButtonProps>(
  ({ variant = "primary", size = "default", className = "", disabled, ...props }, ref) => (
    <button
      ref={ref}
      disabled={disabled}
      className={`inline-flex items-center justify-center gap-2 rounded-[20px] font-bold font-heading tracking-wide transition-all duration-200 focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-clay-accent/30 focus-visible:ring-offset-2 focus-visible:ring-offset-clay-canvas disabled:pointer-events-none disabled:opacity-50 ${
        disabled ? "" : "hover:-translate-y-1 active:translate-y-0 active:scale-[0.92] active:shadow-clayPressed"
      } ${variantClasses[variant]} ${sizeClasses[size]} ${className}`}
      {...props}
    />
  ),
);
ClayButton.displayName = "ClayButton";
