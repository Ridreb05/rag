export function Background() {
  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden" aria-hidden="true">
      <div className="absolute -top-[10%] -left-[10%] h-[60vh] w-[60vh] animate-clay-float rounded-full bg-[#8B5CF6]/10 blur-3xl" />
      <div className="absolute -right-[10%] top-[15%] h-[55vh] w-[55vh] animate-clay-float-delayed rounded-full bg-[#EC4899]/10 blur-3xl [animation-delay:2s]" />
      <div className="absolute bottom-[5%] left-[15%] h-[50vh] w-[50vh] animate-clay-float-slow rounded-full bg-[#0EA5E9]/10 blur-3xl [animation-delay:4s]" />
      <div className="absolute bottom-[-10%] right-[5%] h-[45vh] w-[45vh] animate-clay-float rounded-full bg-[#10B981]/10 blur-3xl [animation-delay:3s]" />
    </div>
  );
}
