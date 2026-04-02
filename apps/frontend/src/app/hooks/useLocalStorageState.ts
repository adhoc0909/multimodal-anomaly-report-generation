// src/app/hooks/useLocalStorageState.ts
import { useEffect, useMemo, useState } from "react";

type Options<T> = {
  serialize?: (v: T) => string;
  deserialize?: (raw: string) => T;
  normalize?: (v: T) => T; // clamp 등
};

export function useLocalStorageState<T>(
  key: string,
  initial: T,
  options?: Options<T>
) {
  const serialize = options?.serialize ?? ((v: T) => JSON.stringify(v));
  const deserialize =
    options?.deserialize ??
    ((raw: string) => {
      try {
        return JSON.parse(raw) as T;
      } catch {
        return initial;
      }
    });

  const [value, setValue] = useState<T>(() => {
    const raw = localStorage.getItem(key);
    if (!raw) return options?.normalize ? options.normalize(initial) : initial;
    const v = deserialize(raw);
    return options?.normalize ? options.normalize(v) : v;
  });

  useEffect(() => {
    try {
      localStorage.setItem(key, serialize(value));
    } catch {
    }
  }, [key, value, serialize]);

  return useMemo(() => [value, setValue] as const, [value]);
}
