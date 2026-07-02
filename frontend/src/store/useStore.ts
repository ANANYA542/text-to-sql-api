import { create } from 'zustand'

export type TabType = 'dashboard' | 'generate' | 'pipeline' | 'ml' | 'metrics' | 'experiments' | 'logs' | 'settings';

export interface PipelineStatus {
  step: 'idle' | 'expand' | 'retrieve' | 'rerank' | 'ml' | 'generate' | 'validate' | 'complete' | 'failed';
  message: string;
  timeMs?: Record<string, number>;
}

interface AppState {
  activeTab: TabType;
  setActiveTab: (tab: TabType) => void;
  
  // Query State
  currentQuestion: string;
  setCurrentQuestion: (q: string) => void;
  pipelineStatus: PipelineStatus;
  setPipelineStatus: (status: Partial<PipelineStatus>) => void;
  executionResult: any | null;
  setExecutionResult: (res: any) => void;
  
  // Retrieval Pipeline Page State
  selectedTable: string | null;
  setSelectedTable: (table: string | null) => void;
  
  // SQL Workspace State
  sqlWorkspaceQuery: string;
  setSqlWorkspaceQuery: (sql: string) => void;
  sqlExecutionResult: any | null;
  setSqlExecutionResult: (res: any) => void;
  sqlValidationResult: any | null;
  setSqlValidationResult: (res: any) => void;
  
  // Telemetry Cache
  healthStatus: any | null;
  setHealthStatus: (health: any) => void;

  // Settings parameters
  hybridAlpha: number;
  setHybridAlpha: (alpha: number) => void;
  confidenceThreshold: number;
  setConfidenceThreshold: (threshold: number) => void;
  maxTables: number;
  setMaxTables: (count: number) => void;
  activeModel: string;
  setActiveModel: (model: string) => void;
}

export const useStore = create<AppState>((set) => ({
  activeTab: 'dashboard',
  setActiveTab: (tab) => set({ activeTab: tab }),
  
  currentQuestion: '',
  setCurrentQuestion: (q) => set({ currentQuestion: q }),
  pipelineStatus: { step: 'idle', message: '' },
  setPipelineStatus: (status) => set((state) => ({ pipelineStatus: { ...state.pipelineStatus, ...status } })),
  executionResult: null,
  setExecutionResult: (res) => set({ executionResult: res }),
  
  selectedTable: null,
  setSelectedTable: (table) => set({ selectedTable: table }),
  
  sqlWorkspaceQuery: '',
  setSqlWorkspaceQuery: (sql) => set({ sqlWorkspaceQuery: sql }),
  sqlExecutionResult: null,
  setSqlExecutionResult: (res) => set({ sqlExecutionResult: res }),
  sqlValidationResult: null,
  setSqlValidationResult: (res) => set({ sqlValidationResult: res }),
  
  healthStatus: null,
  setHealthStatus: (health) => set({ healthStatus: health }),

  hybridAlpha: 0.6,
  setHybridAlpha: (alpha) => set({ hybridAlpha: alpha }),
  confidenceThreshold: 0.85,
  setConfidenceThreshold: (threshold) => set({ confidenceThreshold: threshold }),
  maxTables: 5,
  setMaxTables: (count) => set({ maxTables: count }),
  activeModel: 'Llama 3.1 70B',
  setActiveModel: (model) => set({ activeModel: model }),
}))
