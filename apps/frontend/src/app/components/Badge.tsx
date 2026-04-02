// src/app/components/Badge.tsx
import React from 'react';

interface BadgeProps {
  children: React.ReactNode;
  variant: 'OK' | 'NG' | 'REVIEW' | 'low' | 'med' | 'high' | 'spike' | 'pattern' | 'line_issue';
  size?: 'sm' | 'md';
}

export function Badge({ children, variant, size = 'md' }: BadgeProps) {
  const baseClasses = 'inline-flex items-center justify-center rounded font-medium whitespace-nowrap leading-none';

  const sizeClasses = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-3 py-1 text-sm'
  };

  const variantClasses = {
    OK: 'bg-green-100 text-green-800 border border-green-200',
    NG: 'bg-red-100 text-red-800 border border-red-200',
    REVIEW: 'bg-amber-100 text-amber-800 border border-amber-200',
    low: 'bg-gray-100 text-gray-700 border border-gray-200',
    med: 'bg-orange-100 text-orange-700 border border-orange-200',
    high: 'bg-red-100 text-red-700 border border-red-200',
    spike: 'bg-red-50 text-red-700',
    pattern: 'bg-amber-50 text-amber-700',
    line_issue: 'bg-orange-50 text-orange-700'
  };

  return (
    <span className={`${baseClasses} ${sizeClasses[size]} ${variantClasses[variant]}`}>
      {children}
    </span>
  );
}
