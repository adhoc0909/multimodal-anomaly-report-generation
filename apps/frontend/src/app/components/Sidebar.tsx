// src/app/components/Sidebar.tsx
import React from "react";
import {
  LayoutDashboard,
  ListFilter,
  FileText,
  Settings,
} from "lucide-react";

interface SidebarProps {
  currentPage: string;
  onNavigate: (page: string) => void;
}

export function Sidebar({ currentPage, onNavigate }: SidebarProps) {
  const navItems = [
    { id: "overview", label: "개요", icon: LayoutDashboard },
    { id: "queue", label: "이상 큐", icon: ListFilter },
    { id: "report", label: "AI 리포트", icon: FileText }, 
    { id: "settings", label: "설정", icon: Settings },
  ];

  return (
    <div className="w-64 bg-gray-900 text-white h-screen flex flex-col sticky top-0">
      <div className="p-6 border-b border-gray-800">
        <h1 className="text-xl font-semibold">품질 관리 시스템</h1>
      </div>

      <nav className="flex-1 p-4">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentPage === item.id;

          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg mb-2 transition-colors ${
                isActive
                  ? "bg-blue-600 text-white"
                  : "text-gray-300 hover:bg-gray-800"
              }`}
            >
              <Icon className="w-5 h-5" />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="p-4 border-t border-gray-800">
        <div className="flex items-center gap-3 px-4 py-3">
          <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center">
            <span className="text-sm font-medium">이</span>
          </div>
          <div>
            <p className="text-sm font-medium">이호욱</p>
            <p className="text-xs text-gray-400">품질관리팀</p>
          </div>
        </div>
      </div>
    </div>
  );
}
