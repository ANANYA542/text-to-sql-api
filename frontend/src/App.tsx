import React, { useEffect } from 'react'
import { useStore } from './store/useStore'
import type { TabType } from './store/useStore'
import Dashboard from './components/Dashboard'
import PipelinePage from './components/PipelinePage'
import SQLWorkspace from './components/SQLWorkspace'
import MLDashboard from './components/MLDashboard'
import ExperimentTracker from './components/ExperimentTracker'
import LogsPage from './components/LogsPage'
import MetricsPage from './components/MetricsPage'
import { 
  LayoutDashboard, 
  Search, 
  Code2, 
  Brain, 
  FlaskConical, 
  FileText, 
  TrendingUp,
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
      case 'pipeline':
        return <PipelinePage />
      case 'sql':
        return <SQLWorkspace />
      case 'ml':
        return <MLDashboard />
      case 'experiments':
        return <ExperimentTracker />
      case 'logs':
        return <LogsPage />
      case 'metrics':
        return <MetricsPage />
      default:
        return <Dashboard />
    }
  }

  const navItems = [
    { id: 'dashboard' as TabType, label: 'Workspace', icon: LayoutDashboard },
    { id: 'pipeline' as TabType, label: 'Retrieval Funnel', icon: Search },
    { id: 'sql' as TabType, label: 'SQL Editor', icon: Code2 },
    { id: 'ml' as TabType, label: 'Feature Space', icon: Brain },
    { id: 'experiments' as TabType, label: 'CV Experiments', icon: FlaskConical },
    { id: 'logs' as TabType, label: 'Structured Logs', icon: FileText },
    { id: 'metrics' as TabType, label: 'System Analytics', icon: TrendingUp },
  ]

  return (
    <div className="flex min-h-screen bg-[#FAF8F2] text-[#1F2421] selection:bg-[#5A738E]/10 select-none antialiased">
      {/* ── Left Sidebar ────────────────────────────────────────── */}
      <aside className="w-64 border-r border-[#EEE7DA] bg-[#F5F1E8]/70 backdrop-blur-md flex flex-col justify-between">
        <div className="p-6">
          <div className="flex items-center gap-3 mb-8">
            <div className="w-8 h-8 rounded-lg bg-[#5A738E] flex items-center justify-center text-[#FAF8F2] font-semibold text-sm shadow-sm">
              T
            </div>
            <div>
              <h1 className="font-semibold text-[#1F2421] text-sm tracking-tight">Text-to-SQL</h1>
              <p className="text-[10px] text-[#8E9490] uppercase tracking-wider font-medium">Enterprise Engine</p>
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
                  className={`w-full flex items-center gap-3 px-3 py-2 text-xs font-medium rounded-md transition-all ${
                    isActive 
                      ? 'bg-[#5A738E] text-[#FAF8F2] shadow-sm' 
                      : 'text-[#5C625E] hover:text-[#1F2421] hover:bg-[#EEE7DA]/50'
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
        <div className="p-6 border-t border-[#EEE7DA]">
          <div className="flex items-center gap-2">
            {healthStatus?.engine_loaded ? (
              <>
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#6E8B7E] opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-[#6E8B7E]"></span>
                </span>
                <span className="text-[10px] text-[#5C625E] font-medium tracking-wide">
                  Active (97 Tables Connected)
                </span>
              </>
            ) : healthStatus?.error ? (
              <>
                <AlertCircle className="w-3.5 h-3.5 text-[#A0522D]" />
                <span className="text-[10px] text-[#A0522D] font-medium tracking-wide truncate max-w-[170px]" title={healthStatus.error}>
                  {healthStatus.error}
                </span>
              </>
            ) : (
              <>
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#C19A6B] opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-[#C19A6B]"></span>
                </span>
                <span className="text-[10px] text-[#C19A6B] font-medium tracking-wide">
                  Connecting to API...
                </span>
              </>
            )}
          </div>
        </div>
      </aside>

      {/* ── Main Workspace ──────────────────────────────────────── */}
      <main className="flex-1 flex flex-col min-w-0 overflow-y-auto">
        <header className="h-16 border-b border-[#EEE7DA] px-8 flex items-center justify-between bg-[#FAF8F2]/40 backdrop-blur-md sticky top-0 z-10">
          <div className="flex items-center gap-2 text-xs font-semibold text-[#8E9490]">
            <span>System</span>
            <span>/</span>
            <span className="text-[#1F2421] capitalize">{activeTab}</span>
          </div>
          <div className="flex items-center gap-4 text-xs font-medium text-[#5C625E]">
            {healthStatus?.engine_loaded && (
              <span className="bg-[#EEE7DA] px-2 py-0.5 rounded text-[10px] text-[#1F2421]">
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
