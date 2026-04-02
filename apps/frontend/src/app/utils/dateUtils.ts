// src/utils/dateUtils.ts

export function startOfDay(d: Date) {
  const x = new Date(d);
  x.setHours(0, 0, 0, 0);
  return x;
}

export function endOfDay(d: Date) {
  const x = new Date(d);
  x.setHours(23, 59, 59, 999);
  return x;
}

export function getDateRangeWindow(range: string, now = new Date()): { from: Date; to: Date } | null {
  const todayStart = startOfDay(now);
  const todayEnd = endOfDay(now);

  if (range === 'today') return { from: todayStart, to: todayEnd };

  if (range === 'yesterday') {
    const y = new Date(todayStart);
    y.setDate(y.getDate() - 1);
    return { from: startOfDay(y), to: endOfDay(y) };
  }

  if (range === 'week') {
    const from = new Date(todayStart);
    from.setDate(from.getDate() - 6);
    return { from, to: todayEnd };
  }

  if (range === 'month') {
    const from = new Date(todayStart);
    from.setDate(from.getDate() - 29);
    return { from, to: todayEnd };
  }

  return null;
}
