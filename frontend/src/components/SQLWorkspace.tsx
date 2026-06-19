import React, { useState, useEffect } from 'react'
import { useStore } from '../store/useStore'
import { Play, FileText, Clipboard, Check, RefreshCw, AlertTriangle, Terminal } from 'lucide-react'

const SQLWorkspace: React.FC = () => {
  const { 
    sqlWorkspaceQuery, 
    setSqlWorkspaceQuery, 
    sqlExecutionResult, 
    setSqlExecutionResult,
    sqlValidationResult, 
    setSqlValidationResult,
    executionResult
  } = useStore()

  const [copied, setCopied] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [explainResult, setExplainResult] = useState<any | null>(null)

  // Prefill SQL editor if a query was generated in the dashboard workspace
  useEffect(() => {
    if (executionResult?.sql && !sqlWorkspaceQuery) {
      setSqlWorkspaceQuery(executionResult.sql)
    }
  }, [executionResult, sqlWorkspaceQuery, setSqlWorkspaceQuery])

  const handleCopy = () => {
    if (!sqlWorkspaceQuery) return
    navigator.clipboard.writeText(sqlWorkspaceQuery)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const runQuery = async () => {
    if (!sqlWorkspaceQuery.trim()) return
    setLoading(true)
    setError(null)
    setExplainResult(null)
    setSqlExecutionResult(null)

    try {
      const res = await fetch('/api/execute-sql', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sql: sqlWorkspaceQuery })
      })

      const data = await res.json()
      if (res.ok) {
        setSqlExecutionResult(data)
        setSqlValidationResult({ valid: true })
      } else {
        setError(data.detail || 'SQL Query execution failed')
        setSqlValidationResult({ valid: false, error: data.detail })
      }
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const explainPlan = async () => {
    if (!sqlWorkspaceQuery.trim()) return
    setLoading(true)
    setError(null)
    setExplainResult(null)
    setSqlExecutionResult(null)

    try {
      const res = await fetch('/api/explain-sql', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sql: sqlWorkspaceQuery })
      })

      const data = await res.json()
      if (res.ok) {
        setExplainResult(data)
      } else {
        setError(data.detail || 'SQL Explain execution failed')
      }
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div>
        <h2 className="text-2xl font-semibold tracking-tight text-[#1F2421]">SQL Workspace</h2>
        <p className="text-xs text-[#5C625E] mt-1">
          Directly execute queries and inspect query execution plans against the SQLite dw schema database.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Side: SQL Code Editor */}
        <div className="lg:col-span-2 space-y-4">
          <div className="bg-[#F5F1E8]/40 border border-[#EEE7DA] rounded-lg p-6 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-[#8E9490]">SQL Query Editor</span>
              <div className="flex items-center gap-2">
                <button
                  onClick={handleCopy}
                  className="text-[#5C625E] hover:text-[#1F2421] transition-all p-1 hover:bg-[#EEE7DA]/50 rounded cursor-pointer"
                  title="Copy Query"
                >
                  {copied ? <Check className="w-4 h-4 text-[#6E8B7E]" /> : <Clipboard className="w-4 h-4" />}
                </button>
              </div>
            </div>

            <textarea
              value={sqlWorkspaceQuery}
              onChange={(e) => setSqlWorkspaceQuery(e.target.value)}
              className="w-full h-80 p-4 bg-[#FAF8F2] border border-[#EEE7DA] rounded-lg font-mono text-xs text-[#1F2421] placeholder-[#8E9490] focus:outline-none focus:border-[#5A738E] transition-all resize-none"
              placeholder="SELECT * FROM SIS_DEPARTMENT LIMIT 10;"
            />

            <div className="flex items-center gap-3">
              <button
                onClick={runQuery}
                disabled={loading || !sqlWorkspaceQuery.trim()}
                className="bg-[#5A738E] hover:bg-[#43586F] text-[#FAF8F2] font-medium text-xs px-4 py-2.5 rounded transition-all flex items-center gap-2 cursor-pointer shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                Run Query
              </button>
              <button
                onClick={explainPlan}
                disabled={loading || !sqlWorkspaceQuery.trim()}
                className="bg-[#FAF8F2] hover:bg-[#EEE7DA]/50 text-[#5C625E] border border-[#EEE7DA] font-medium text-xs px-4 py-2.5 rounded transition-all flex items-center gap-2 cursor-pointer"
              >
                <FileText className="w-3.5 h-3.5" />
                Explain Plan
              </button>
            </div>
          </div>
        </div>

        {/* Right Side: Validation & Information */}
        <div className="space-y-4">
          <div className="bg-[#F5F1E8]/40 border border-[#EEE7DA] rounded-lg p-6 space-y-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-[#8E9490]">Query Details</h3>

            <div className="space-y-3 text-xs">
              <div className="flex items-center justify-between border-b border-[#EEE7DA] pb-2">
                <span className="text-[#8E9490]">Source Database</span>
                <span className="font-mono text-[#5C625E] font-medium">beaver_dw.db</span>
              </div>
              <div className="flex items-center justify-between border-b border-[#EEE7DA] pb-2">
                <span className="text-[#8E9490]">Engine Mode</span>
                <span className="font-mono text-[#5C625E] font-medium">read-only SELECT</span>
              </div>
            </div>

            {sqlValidationResult && (
              <div className="pt-2">
                <span className="text-[10px] uppercase font-semibold text-[#8E9490] tracking-wider block mb-1.5">Validation Check</span>
                {sqlValidationResult.valid ? (
                  <div className="bg-[#6E8B7E]/10 border border-[#6E8B7E]/30 text-[#6E8B7E] p-3.5 rounded text-xs">
                    ✓ Query syntax is valid.
                  </div>
                ) : (
                  <div className="bg-[#A0522D]/10 border border-[#A0522D]/30 text-[#A0522D] p-3.5 rounded text-xs leading-relaxed">
                    ✕ {sqlValidationResult.error || 'Invalid query structure.'}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Execution / Explanation Results Output Grid */}
      {(sqlExecutionResult || explainResult || error || loading) && (
        <div className="bg-[#F5F1E8]/40 border border-[#EEE7DA] rounded-lg p-6 space-y-4">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-[#8E9490] flex items-center gap-2">
            <Terminal className="w-3.5 h-3.5" />
            Execution Console
          </h3>

          {loading && (
            <div className="flex items-center justify-center py-12 text-xs text-[#5C625E] gap-2">
              <RefreshCw className="w-4 h-4 animate-spin text-[#5A738E]" />
              Query processing...
            </div>
          )}

          {error && (
            <div className="bg-[#A0522D]/10 border border-[#A0522D]/40 p-4 rounded text-xs text-[#A0522D] flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
              <div>
                <span className="font-semibold block mb-0.5">Execution Failed</span>
                <pre className="font-mono whitespace-pre-wrap break-all text-[10px]">{error}</pre>
              </div>
            </div>
          )}

          {sqlExecutionResult && !loading && (
            <div className="overflow-x-auto border border-[#EEE7DA] rounded-lg max-h-96">
              <table className="w-full text-left text-xs border-collapse">
                <thead className="bg-[#FAF8F2] sticky top-0 border-b border-[#EEE7DA]">
                  <tr>
                    {sqlExecutionResult.columns.map((col: string, idx: number) => (
                      <th key={idx} className="p-3 font-semibold text-[#1F2421] tracking-tight whitespace-nowrap">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="bg-[#FAF8F2]/40 divide-y divide-[#EEE7DA]/50">
                  {sqlExecutionResult.rows.length === 0 ? (
                    <tr>
                      <td colSpan={sqlExecutionResult.columns.length} className="p-8 text-center text-[#8E9490]">
                        Query execution returned 0 rows.
                      </td>
                    </tr>
                  ) : (
                    sqlExecutionResult.rows.map((row: any[], rowIdx: number) => (
                      <tr key={rowIdx} className="hover:bg-[#EEE7DA]/20 transition-all">
                        {row.map((val: any, colIdx: number) => (
                          <td key={colIdx} className="p-3 text-[#5C625E] font-mono text-[11px] whitespace-nowrap max-w-[250px] truncate" title={String(val)}>
                            {val === null ? <span className="text-[#8E9490] italic">null</span> : String(val)}
                          </td>
                        ))}
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          )}

          {explainResult && !loading && (
            <div className="overflow-x-auto border border-[#EEE7DA] rounded-lg max-h-96">
              <table className="w-full text-left text-xs border-collapse">
                <thead className="bg-[#FAF8F2] sticky top-0 border-b border-[#EEE7DA]">
                  <tr>
                    {explainResult.columns.map((col: string, idx: number) => (
                      <th key={idx} className="p-3 font-semibold text-[#1F2421] tracking-tight whitespace-nowrap">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="bg-[#FAF8F2]/40 divide-y divide-[#EEE7DA]/50">
                  {explainResult.rows.map((row: any[], rowIdx: number) => (
                    <tr key={rowIdx} className="hover:bg-[#EEE7DA]/20 transition-all font-mono text-[11px]">
                      {row.map((val: any, colIdx: number) => (
                        <td key={colIdx} className="p-3 text-[#5C625E] whitespace-nowrap">
                          {String(val)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default SQLWorkspace
