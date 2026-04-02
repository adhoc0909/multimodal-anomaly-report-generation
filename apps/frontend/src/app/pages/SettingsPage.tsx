// src/app/pages/SettingsPage.tsx
import React, { useState } from "react";
import {
  Users,
  Bell,
  Database,
  Key,
  LogOut,
  CheckCircle,
} from "lucide-react";
import type { NotificationSettings } from "../data/AlertData";

interface SettingsPageProps {
  activeModel: string;
  onModelChange: (model: string) => void;
  llmModels: string[];
  modelSyncing: boolean;

  notifications: NotificationSettings;
  onNotificationsChange: (next: NotificationSettings) => void;

  apiKey: string;
  onApiKeyChange: (key: string) => void;
}

export function SettingsPage({
  activeModel,
  onModelChange,
  llmModels,
  modelSyncing,
  notifications,
  onNotificationsChange,
  apiKey,
  onApiKeyChange,
}: SettingsPageProps) {
  const [inputKey, setInputKey] = useState("");

  // 현재 선택된 모델이 Gemini 계열인지 확인
  const isGeminiModel = activeModel.toLowerCase().includes("gemini");

  const handleLogin = () => {
    if (inputKey.trim()) {
      onApiKeyChange(inputKey.trim());
      setInputKey("");
      alert("API Key가 등록되었습니다");
    }
  };

  const handleLogout = () => {
    if (confirm("API Key가 삭제됩니다.")) {
      onApiKeyChange("");
    }
  };

  return (
    <div className="p-8">
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-gray-900 mb-2">설정</h1>
        <p className="text-sm text-gray-600">시스템 설정 및 환경 구성</p>
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* Model Settings */}
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-blue-100 rounded-lg">
              <Database className="w-5 h-5 text-blue-600" />
            </div>
            <h2 className="text-lg font-medium text-gray-900">모델 설정</h2>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                활성 LLM 모델
              </label>
              <select
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={activeModel}
                onChange={(e) => onModelChange(e.target.value)}
                disabled={modelSyncing}
              >
                {(llmModels || []).map((modelName) => (
                  <option key={modelName} value={modelName}>
                    {modelName}
                  </option>
                ))}
              </select>
              <p className="text-xs text-gray-500 mt-1">
                변경 시 백엔드 LLM 모델이 즉시 전환됩니다.
              </p>
            </div>
          </div>
        </div>

        {/* Notification Settings */}
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-amber-100 rounded-lg">
              <Bell className="w-5 h-5 text-amber-600" />
            </div>
            <h2 className="text-lg font-medium text-gray-900">알림 설정</h2>
          </div>

          <div className="space-y-3">
            <label className="flex items-center gap-3">
              <input
                type="checkbox"
                checked={notifications.highSeverity}
                onChange={(e) =>
                  onNotificationsChange({
                    ...notifications,
                    highSeverity: e.target.checked,
                  })
                }
                className="w-4 h-4 text-blue-600 rounded"
              />
              <span className="text-sm text-gray-700">
                심각한 결함(High Severity) 감지 시 알림
              </span>
            </label>

            <label className="flex items-center gap-3">
              <input
                type="checkbox"
                checked={notifications.consecutiveDefects}
                onChange={(e) =>
                  onNotificationsChange({
                    ...notifications,
                    consecutiveDefects: e.target.checked,
                  })
                }
                className="w-4 h-4 text-blue-600 rounded"
              />
              <span className="text-sm text-gray-700">연속 불량 감지 알림</span>
            </label>

            <label className="flex items-center gap-3">
              <input
                type="checkbox"
                checked={notifications.dailyReport}
                onChange={(e) =>
                  onNotificationsChange({
                    ...notifications,
                    dailyReport: e.target.checked,
                  })
                }
                className="w-4 h-4 text-blue-600 rounded"
              />
              <span className="text-sm text-gray-700">일일 리포트 생성 완료 알림</span>
            </label>

            <label className="flex items-center gap-3">
              <input
                type="checkbox"
                checked={notifications.systemError}
                onChange={(e) =>
                  onNotificationsChange({
                    ...notifications,
                    systemError: e.target.checked,
                  })
                }
                className="w-4 h-4 text-blue-600 rounded"
              />
              <span className="text-sm text-gray-700">시스템 오류 및 장애 알림</span>
            </label>
          </div>
        </div>

        {/* User Management */}
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-green-100 rounded-lg">
              <Users className="w-5 h-5 text-green-600" />
            </div>
            <h2 className="text-lg font-medium text-gray-900">사용자 관리</h2>
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <div>
                <p className="text-sm font-medium text-gray-900">손우정</p>
                <p className="text-xs text-gray-500">품질관리팀 · 관리자</p>
              </div>
              <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded">활성</span>
            </div>

            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <div>
                <p className="text-sm font-medium text-gray-900">노성호</p>
                <p className="text-xs text-gray-500">품질관리팀 · 검사원</p>
              </div>
              <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded">활성</span>
            </div>

            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <div>
                <p className="text-sm font-medium text-gray-900">문국현</p>
                <p className="text-xs text-gray-500">생산팀 · 검사원</p>
              </div>
              <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded">활성</span>
            </div>
          </div>
        </div>

        {/* API Key Management */}
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-purple-100 rounded-lg">
              <Key className="w-5 h-5 text-purple-600" />
            </div>
            <h2 className="text-lg font-medium text-gray-900">
              API Key 관리
            </h2>
          </div>

          {/* Gemini 모델이 아닐 경우 안내 메시지 출력 */}
          {!isGeminiModel ? (
            <div className="flex flex-col items-center justify-center py-6 text-gray-500">
              <Key className="w-8 h-8 mb-2 opacity-30" />
              <p className="text-sm font-medium">로컬 모델 사용 중</p>
              <p className="text-xs mt-1">현재 선택된 모델은 API Key가 필요하지 않습니다.</p>
            </div>
          ) : apiKey ? (
            <div className="space-y-4">
              <div className="p-4 bg-white border border-green-600 rounded-lg flex items-center gap-3">
                <CheckCircle className="w-5 h-5 text-green-600" />
                <div>
                  <p className="text-sm font-medium text-green-700">
                    등록된 API Key
                  </p>
                  <p className="text-xs text-green-600 mt-1">
                    API Key: {apiKey.length > 12 
                      ? `${apiKey.substring(0, 8)}...${apiKey.substring(apiKey.length - 4)}` 
                      : "******"}
                  </p>
                </div>
              </div>

              <button
                onClick={handleLogout}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 border border-red-200 text-red-700 rounded-lg hover:bg-red-50 transition-colors"
              >
                <LogOut className="w-4 h-4" />
                <span>Key 삭제</span>
              </button>
            </div>
          ) : (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  API Key 입력
                </label>
                <input
                  type="password"
                  value={inputKey}
                  onChange={(e) => setInputKey(e.target.value)}
                  placeholder="sk-..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <p className="text-xs text-gray-500 mt-1">
                  발급받은 API Key를 입력하여 로그인하세요.
                </p>
              </div>

              <button
                onClick={handleLogin}
                disabled={!inputKey.trim()}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Key className="w-4 h-4" />
                <span>API Key 등록</span>
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}