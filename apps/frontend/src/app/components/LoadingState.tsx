// src/app/components/LoadingState.tsx

//깔끔한 스피너 + 상태 문구
/*import React from "react";

export function LoadingState({ message = "데이터 로딩 중..." }: { message?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-20">
      <div className="w-12 h-12 rounded-full border-4 border-gray-200 border-t-blue-600 animate-spin" />
      <p className="mt-4 text-sm text-gray-600">{message}</p>
      <p className="mt-1 text-xs text-gray-400">서버에서 최신 데이터를 불러오고 있습니다.</p>
    </div>
  );
}*/

// 화살표가 돌아가는 로딩 화면
import React from "react";
import { RefreshCw } from "lucide-react";

export function LoadingState({
  title = "불러오는 중",
  message = "품질 검사 데이터를 가져오고 있습니다.",
}: {
  title?: string;
  message?: string;
}) {
  return (
    <div className="flex flex-col items-center justify-center py-20">
      <div className="flex items-center gap-3">
        <RefreshCw className="w-6 h-6 text-blue-600 animate-spin" />
        <span className="text-base font-semibold text-gray-900">{title}</span>
      </div>

      <p className="mt-2 text-sm text-gray-600">{message}</p>

      <div className="mt-6 w-[280px] h-2 bg-gray-200 rounded-full overflow-hidden">
        <div className="h-full w-1/3 bg-blue-600 rounded-full animate-pulse" />
      </div>

      <p className="mt-2 text-xs text-gray-400">시간이 걸릴 수 있습니다.</p>
    </div>
  );
}


