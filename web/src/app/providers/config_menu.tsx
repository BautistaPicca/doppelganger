"use client";

import { createContext, useContext, useState, Dispatch, SetStateAction } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { usePathname } from "next/navigation";

interface ConfigMenuContextType {
  open: boolean;
  setOpen: Dispatch<SetStateAction<boolean>>;
  chooseConfig: (mode: string) => Promise<void>;
}

const ConfigMenuContext = createContext<ConfigMenuContextType | null>(null);

export const useConfigMenu = () => {
  const ctx = useContext(ConfigMenuContext);
  if (!ctx) throw new Error("useConfigMenu debe usarse dentro de ConfigMenuProvider");
  return ctx;
};

export const ConfigMenuProvider = ({ children }: { children: React.ReactNode }) => {
  const [open, setOpen] = useState(false);
  const pathname = usePathname();
  const match = pathname.endsWith("match");
  const chooseConfig = async (mode: string) => {
    await fetch("http://localhost:5000/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
    setOpen(false);
  };

  return (
    <ConfigMenuContext.Provider value={{ open, setOpen, chooseConfig }}>
      {children}

      <button
        onClick={() => setOpen(true)}
        className="fixed bottom-6 right-6 z-50 p-4 rounded-full shadow-xl bg-blue-600 text-white hover:bg-blue-700 transition"
      >
        ⚙️
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            className="fixed inset-0 bg-black/40 backdrop-blur-sm z-50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setOpen(false)}
          >
            <motion.div
              className="absolute right-0 top-0 h-full w-80 bg-white shadow-xl p-6 flex flex-col"
              initial={{ x: 300 }}
              animate={{ x: 0 }}
              exit={{ x: 300 }}
              onClick={(e) => e.stopPropagation()}
            >
              {!match && <><h2 className="text-xl font-bold mb-6">Configuración del Sistema</h2>

              <button
                onClick={() => chooseConfig("pretrained")}
                className="p-3 mb-3 rounded-lg border hover:bg-gray-100"
              >
                Pre-entrenado
              </button>

              <button
                onClick={() => chooseConfig("trained")}
                className="p-3 mb-3 rounded-lg border hover:bg-gray-100"
              >
                Entrenado
              </button>

              <button
                onClick={() => chooseConfig("backbone_trained")}
                className="p-3 mb-3 rounded-lg border hover:bg-gray-100"
              >
                Backbone + Entrenado
              </button>

              <div className="mt-auto text-center text-sm text-gray-500">
                Selecciona una configuración.
              </div></>}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </ConfigMenuContext.Provider>
  );
};
