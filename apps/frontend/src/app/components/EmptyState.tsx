// src/app/components/EmptyState.tsx
import React from 'react';
import { FileX, AlertCircle } from 'lucide-react';

interface EmptyStateProps {
  title: string;
  description: string;
  type?: 'empty' | 'error';
}

export function EmptyState({ title, description, type = 'empty' }: EmptyStateProps) {
  const Icon = type === 'error' ? AlertCircle : FileX;
  
  return (
    <div className="flex flex-col items-center justify-center py-16 px-4">
      <div className={`p-4 rounded-full mb-4 ${
        type === 'error' ? 'bg-red-50' : 'bg-gray-50'
      }`}>
        <Icon className={`w-12 h-12 ${
          type === 'error' ? 'text-red-400' : 'text-gray-400'
        }`} />
      </div>
      <h3 className="text-lg font-medium text-gray-900 mb-2">{title}</h3>
      <p className="text-sm text-gray-500 text-center max-w-md">{description}</p>
    </div>
  );
}
