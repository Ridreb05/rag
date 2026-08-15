import * as Toast from "@radix-ui/react-toast";
import { X } from "lucide-react";

export function ToastLayer({ message, onOpenChange }: { message: string | null; onOpenChange: (open: boolean) => void }) {
  return (
    <>
      <Toast.Root
        className="grid grid-cols-[1fr_auto] items-start gap-x-4 gap-y-1 rounded-[24px] bg-white p-5 shadow-clayCard data-[state=open]:animate-in data-[state=open]:slide-in-from-bottom-4 data-[state=closed]:animate-out data-[state=closed]:fade-out"
        open={Boolean(message)}
        onOpenChange={onOpenChange}
      >
        <Toast.Title className="font-heading text-lg font-extrabold text-clay-foreground">Request update</Toast.Title>
        <Toast.Close
          aria-label="Dismiss"
          className="row-span-2 grid h-8 w-8 place-items-center rounded-full text-clay-muted transition-colors hover:bg-clay-accent/10 hover:text-clay-accent"
        >
          <X size={16} />
        </Toast.Close>
        <Toast.Description className="col-start-1 font-body text-sm leading-relaxed text-clay-muted">
          {message}
        </Toast.Description>
      </Toast.Root>
      <Toast.Viewport className="fixed bottom-6 right-6 z-30 m-0 flex w-[min(390px,calc(100vw-3rem))] list-none flex-col gap-3 p-0 outline-none" />
    </>
  );
}
