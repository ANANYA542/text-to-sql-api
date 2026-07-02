import React, { useEffect } from 'react'
import { useStore } from './store/useStore'
import type { TabType } from './store/useStore'
import Dashboard from './components/Dashboard'
import GenerateSQL from './components/GenerateSQL'
import PipelinePage from './components/PipelinePage'
import MLDashboard from './components/MLDashboard'
import ExperimentTracker from './components/ExperimentTracker'
import LogsPage from './components/LogsPage'
import MetricsPage from './components/MetricsPage'
import SettingsPage from './components/SettingsPage'
import { 
  LayoutDashboard, 
  Sparkles, 
  Search, 
  Brain, 
  TrendingUp, 
  FlaskConical, 
  FileText, 
  Sliders, 
  AlertCircle 
} from 'lucide-react'

const App: React.FC = () => {
  const { activeTab, setActiveTab, healthStatus, setHealthStatus } = useStore()

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const res = await fetch('/health')
        if (res.ok) {
          const data = await res.json()
          setHealthStatus(data)
        } else {
          setHealthStatus({ status: 'unhealthy', error: `HTTP ${res.status}` })
        }
      } catch (err: any) {
        setHealthStatus({ status: 'offline', error: err.message })
      }
    }

    fetchHealth()
    const interval = setInterval(fetchHealth, 10000)
    return () => clearInterval(interval)
  }, [setHealthStatus])

  const renderActiveTab = () => {
    switch (activeTab) {
      case 'dashboard':
        return <Dashboard />
      case 'generate':
        return <GenerateSQL />
      case 'pipeline':
        return <PipelinePage />
      case 'ml':
        return <MLDashboard />
      case 'metrics':
        return <MetricsPage />
      case 'experiments':
        return <ExperimentTracker />
      case 'logs':
        return <LogsPage />
      case 'settings':
        return <SettingsPage />
      default:
        return <Dashboard />
    }
  }

  const navItems = [
    { id: 'dashboard' as TabType, label: 'Dashboard', icon: LayoutDashboard },
    { id: 'generate' as TabType, label: 'Generate SQL', icon: Sparkles },
    { id: 'pipeline' as TabType, label: 'Retrieval Pipeline', icon: Search },
    { id: 'ml' as TabType, label: 'ML Ranker', icon: Brain },
    { id: 'metrics' as TabType, label: 'Metrics', icon: TrendingUp },
    { id: 'experiments' as TabType, label: 'Experiment Tracker', icon: FlaskConical },
    { id: 'logs' as TabType, label: 'Logs', icon: FileText },
    { id: 'settings' as TabType, label: 'Settings', icon: Sliders },
  ]

  return (
    <div className="flex min-h-screen bg-[#F8FAFC] text-[#0F172A] selection:bg-[#2563EB]/10 select-none antialiased font-sans">
      {/* ── Left Sidebar ────────────────────────────────────────── */}
      <aside className="w-64 border-r border-[#1E293B] bg-[#0B0F19] flex flex-col justify-between shrink-0">
        <div className="p-6">
          <div className="flex items-center gap-3 mb-8">
            <div className="w-8 h-8 rounded-lg bg-[#2563EB] flex items-center justify-center text-white font-semibold text-sm shadow-sm">
              T
            </div>
            <div>
              <h1 className="font-semibold text-white text-sm tracking-tight">Text-to-SQL</h1>
              <p className="text-[10px] text-slate-500 uppercase tracking-wider font-medium">Enterprise Engine</p>
            </div>
          </div>

          <nav className="space-y-1">
            {navItems.map((item) => {
              const Icon = item.icon
              const isActive = activeTab === item.id
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`w-full flex items-center gap-3 px-3 py-2.5 text-xs font-medium rounded-md transition-all ${
                    isActive 
                      ? 'bg-[#2563EB] text-white shadow-sm' 
                      : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
                  }`}
                >
                  <Icon className="w-4 h-4 shrink-0" />
                  {item.label}
                </button>
              )
            })}
          </nav>
        </div>

        {/* System Health Status Indicator */}
        <div className="p-6 border-t border-[#1E293B]">
          <div className="flex items-center gap-2">
            {healthStatus?.engine_loaded ? (
              <>
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#10B981] opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-[#10B981]"></span>
                </span>
                <span className="text-[10px] text-slate-400 font-medium tracking-wide">
                  Active ({healthStatus.num_tables || 97} Tables Connected)
                </span>
              </>
            ) : healthStatus?.error ? (
              <>
                <AlertCircle className="w-3.5 h-3.5 text-[#EF4444]" />
                <span className="text-[10px] text-[#EF4444] font-medium tracking-wide truncate max-w-[170px]" title={healthStatus.error}>
                  {healthStatus.error}
                </span>
              </>
            ) : (
              <>
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#F59E0B] opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-[#F59E0B]"></span>
                </span>
                <span className="text-[10px] text-[#F59E0B] font-medium tracking-wide">
                  Connecting to API...
                </span>
              </>
            )}
          </div>
        </div>
      </aside>

      {/* ── Main Workspace ──────────────────────────────────────── */}
      <main className="flex-1 flex flex-col min-w-0 overflow-y-auto">
        <header className="h-16 border-b border-[#E2E8F0] px-8 flex items-center justify-between bg-white/40 backdrop-blur-md sticky top-0 z-10">
          <div className="flex items-center gap-2 text-xs font-semibold text-slate-400">
            <span>System</span>
            <span>/</span>
            <span className="text-[#0F172A] capitalize">{activeTab}</span>
          </div>
          <div className="flex items-center gap-4 text-xs font-medium text-slate-600">
            {healthStatus?.engine_loaded && (
              <span className="bg-[#F1F5F9] px-2 py-0.5 rounded text-[10px] text-[#0F172A]">
                v2.0.0 (MPS Ready)
              </span>
            )}
          </div>
        </header>

        <div className="flex-1 p-8 max-w-7xl w-full mx-auto">
          {renderActiveTab()}
        </div>
      </main>
    </div>
  )
}

export default App
