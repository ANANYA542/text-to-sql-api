import React, { useState } from 'react'
import { useStore } from '../store/useStore'
import { Sparkles, Clipboard, Check, RefreshCw, AlertTriangle, Cpu, Terminal } from 'lucide-react'

const Dashboard: React.FC = () => {
  const { 
    currentQuestion, 
    setCurrentQuestion, 
    pipelineStatus, 
    setPipelineStatus,
    executionResult, 
    setExecutionResult 
  } = useStore()
  
  const [copied, setCopied] = useState(false)
  const [dbRows, setDbRows] = useState<any[] | null>(null)
  const [dbCols, setDbCols] = useState<string[] | null>(null)
  const [dbError, setDbError] = useState<string | null>(null)
  const [isExecutingDb, setIsExecutingDb] = useState(false)

  const exampleQueries = [
    "Which departments have more than 100 students?",
    "What is the average total units for subjects offered in the fall term of academic year 2022?",
    "List all buildings with more than 50 rooms and their total assignable area."
  ]

  const handleCopy = () => {
    if (!executionResult?.sql) return
    navigator.clipboard.writeText(executionResult.sql)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const runPipeline = async () => {
    if (!currentQuestion.trim()) return
    
    setPipelineStatus({ step: 'expand', message: 'Expanding natural language query terms...' })
    setExecutionResult(null)
    setDbRows(null)
    setDbCols(null)
    setDbError(null)

    try {
      // Simulate sequential progress based on approximate timings before API resolves
      // Step 1: Expand
      await new Promise(r => setTimeout(r, 200))
      
      // Step 2: Retrieve
      setPipelineStatus({ step: 'retrieve', message: 'Running hybrid BM25 + Vector semantic retrieval...' })
      await new Promise(r => setTimeout(r, 250))
      
      // Step 3: Rerank
      setPipelineStatus({ step: 'rerank', message: 'Scoring candidates via Cross-Encoder...' })
      await new Promise(r => setTimeout(r, 300))

      // Step 4: ML
      setPipelineStatus({ step: 'ml', message: 'Ranking features with Random Forest LTR...' })
      
      // Trigger actual API call
      const res = await fetch('/generate-sql', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: currentQuestion })
      })

      if (!res.ok) throw new Error(`Server returned HTTP ${res.status}`)

      const data = await res.json()
      
      // Step 5: SQL Gen
      setPipelineStatus({ step: 'generate', message: 'Optimizing SQL prompt template & invoking LLM...' })
      await new Promise(r => setTimeout(r, 200))

      // Step 6: Validate
      setPipelineStatus({ step: 'validate', message: 'Validating AST parsing & EXPLAIN execution plans...' })
      await new Promise(r => setTimeout(r, 200))

      setPipelineStatus({ 
        step: 'complete', 
        message: 'Pipeline executed successfully.',
        timeMs: data.latency_breakdown
      })
      setExecutionResult(data)

      // Auto-run generated SQL against SQLite if valid syntax
      if (data.is_valid_syntax && data.sql) {
        autoRunSql(data.sql)
      }
    } catch (err: any) {
      setPipelineStatus({ step: 'failed', message: err.message })
      setExecutionResult({ 
        sql: '', 
        retrieved_tables: [], 
        is_valid_syntax: false, 
        parsing_errors: err.message, 
        confidence: 0,
        prompt_used: ''
      })
    }
  }

  const autoRunSql = async (sql: string) => {
    setIsExecutingDb(true)
    setDbError(null)
    try {
      const res = await fetch('/api/execute-sql', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sql })
      })
      const data = await res.json()
      if (res.ok) {
        setDbRows(data.rows)
        setDbCols(data.columns)
      } else {
        setDbError(data.detail || 'SQL Execution failed')
      }
    } catch (err: any) {
      setDbError(err.message)
    } finally {
      setIsExecutingDb(false)
    }
  }

  return (
    <div className="space-y-8">
      {/* Editorial Title */}
      <div>
        <h2 className="text-2xl font-semibold tracking-tight text-[#1F2421]">Workspace</h2>
        <p className="text-xs text-[#5C625E] mt-1">
          Translate natural language queries into executable SQL queries and trace retrieval heuristics.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Side: Input Panel */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-[#F5F1E8]/40 border border-[#EEE7DA] rounded-lg p-6 space-y-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-[#8E9490]">Natural Language Query</h3>
            
            <textarea
              className="w-full h-32 p-4 bg-[#FAF8F2] border border-[#EEE7DA] rounded-lg text-sm text-[#1F2421] placeholder-[#8E9490] focus:outline-none focus:border-[#5A738E] transition-all resize-none font-sans"
              placeholder="Describe what data you want to retrieve from the Beaver database..."
              value={currentQuestion}
              onChange={(e) => setCurrentQuestion(e.target.value)}
            />

            <div className="flex gap-2">
              <button
                onClick={runPipeline}
                disabled={pipelineStatus.step !== 'idle' && pipelineStatus.step !== 'complete' && pipelineStatus.step !== 'failed'}
                className="bg-[#5A738E] hover:bg-[#43586F] text-[#FAF8F2] font-medium text-xs px-4 py-2.5 rounded transition-all flex items-center gap-2 cursor-pointer shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Sparkles className="w-3.5 h-3.5" />
                Generate SQL
              </button>
            </div>

            <div className="space-y-2 pt-2 border-t border-[#EEE7DA]">
              <span className="text-[10px] uppercase font-semibold text-[#8E9490] tracking-wider block">Suggested Queries</span>
              <div className="space-y-1.5">
                {exampleQueries.map((q, idx) => (
                  <button
                    key={idx}
                    onClick={() => setCurrentQuestion(q)}
                    className="w-full text-left text-xs text-[#5C625E] hover:text-[#1F2421] hover:bg-[#EEE7DA]/50 px-2 py-1.5 rounded transition-all truncate"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Pipeline Trace Visualizer */}
          {pipelineStatus.step !== 'idle' && (
            <div className="bg-[#F5F1E8]/40 border border-[#EEE7DA] rounded-lg p-6 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-[#8E9490]">Pipeline Execution Trace</h3>
                <span className="text-[10px] font-mono text-[#5C625E]">
                  {pipelineStatus.step === 'complete' ? 'Success' : pipelineStatus.step === 'failed' ? 'Error' : 'Running'}
                </span>
              </div>

              {/* Progress Flow Blocks */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {[
                  { id: 'expand', label: '1. Query Expansion' },
                  { id: 'retrieve', label: '2. Hybrid Search' },
                  { id: 'rerank', label: '3. Cross-Encoder' },
                  { id: 'ml', label: '4. Learned Ranker' },
                  { id: 'generate', label: '5. SQL Generation' },
                  { id: 'validate', label: '6. AST Validation' }
                ].map((step) => {
                  const stepOrder = ['idle', 'expand', 'retrieve', 'rerank', 'ml', 'generate', 'validate', 'complete', 'failed']
                  const currentIdx = stepOrder.indexOf(pipelineStatus.step)
                  const stepIdx = stepOrder.indexOf(step.id)
                  
                  let stateStyle = 'bg-[#FAF8F2] border-[#EEE7DA] text-[#8E9490]'
                  if (pipelineStatus.step === 'failed' && currentIdx === stepIdx) {
                    stateStyle = 'bg-[#A0522D]/10 border-[#A0522D] text-[#A0522D]'
                  } else if (currentIdx === stepIdx) {
                    stateStyle = 'bg-[#C19A6B]/10 border-[#C19A6B] text-[#C19A6B] animate-pulse'
                  } else if (currentIdx > stepIdx || pipelineStatus.step === 'complete') {
                    stateStyle = 'bg-[#6E8B7E]/10 border-[#6E8B7E]/40 text-[#6E8B7E]'
                  }

                  return (
                    <div key={step.id} className={`border p-3 rounded text-center transition-all ${stateStyle}`}>
                      <div className="text-[10px] font-semibold tracking-tight">{step.label}</div>
                    </div>
                  )
                })}
              </div>

              <div className="text-[11px] text-[#5C625E] font-medium bg-[#FAF8F2] border border-[#EEE7DA] p-3 rounded flex items-center gap-2">
                <Cpu className="w-3.5 h-3.5 text-[#5A738E] shrink-0" />
                <span>{pipelineStatus.message}</span>
              </div>
            </div>
          )}

          {/* Database Results Grid */}
          {(dbRows || dbError || isExecutingDb) && (
            <div className="bg-[#F5F1E8]/40 border border-[#EEE7DA] rounded-lg p-6 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-[#8E9490] flex items-center gap-2">
                  <Terminal className="w-3.5 h-3.5" />
                  Database Query Output (First 100 Rows)
                </h3>
              </div>

              {isExecutingDb && (
                <div className="flex items-center justify-center py-12 text-xs text-[#5C625E] gap-2">
                  <RefreshCw className="w-4 h-4 animate-spin text-[#5A738E]" />
                  Executing query on SQLite beaver_dw.db...
                </div>
              )}

              {dbError && (
                <div className="bg-[#A0522D]/10 border border-[#A0522D]/40 p-4 rounded text-xs text-[#A0522D] flex items-start gap-2">
                  <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                  <div>
                    <span className="font-semibold block mb-0.5">Execution Failed</span>
                    <pre className="font-mono whitespace-pre-wrap break-all text-[10px]">{dbError}</pre>
                  </div>
                </div>
              )}

              {dbRows && dbCols && !isExecutingDb && (
                <div className="overflow-x-auto border border-[#EEE7DA] rounded-lg max-h-96">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead className="bg-[#FAF8F2] sticky top-0 border-b border-[#EEE7DA]">
                      <tr>
                        {dbCols.map((col, idx) => (
                          <th key={idx} className="p-3 font-semibold text-[#1F2421] tracking-tight whitespace-nowrap">
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="bg-[#FAF8F2]/40 divide-y divide-[#EEE7DA]/50">
                      {dbRows.length === 0 ? (
                        <tr>
                          <td colSpan={dbCols.length} className="p-8 text-center text-[#8E9490]">
                            Query execution returned 0 rows.
                          </td>
                        </tr>
                      ) : (
                        dbRows.map((row, rowIdx) => (
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
            </div>
          )}
        </div>

        {/* Right Side: Output SQL & Details Panel */}
        <div className="space-y-6">
          <div className="bg-[#F5F1E8]/40 border border-[#EEE7DA] rounded-lg p-6 space-y-6">
            <div className="flex items-center justify-between border-b border-[#EEE7DA] pb-4">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-[#8E9490]">Output Panel</h3>
              {executionResult?.confidence !== undefined && (
                <span className="bg-[#6E8B7E]/10 border border-[#6E8B7E]/30 text-[#6E8B7E] text-[10px] px-2 py-0.5 rounded font-mono font-medium">
                  Conf: {Math.round(executionResult.confidence * 100)}%
                </span>
              )}
            </div>

            {/* Generated SQL query with syntax-coloring */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[10px] uppercase font-semibold text-[#8E9490] tracking-wider">Generated SQL</span>
                {executionResult?.sql && (
                  <button onClick={handleCopy} className="text-[#5C625E] hover:text-[#1F2421] transition-all cursor-pointer">
                    {copied ? <Check className="w-3.5 h-3.5 text-[#6E8B7E]" /> : <Clipboard className="w-3.5 h-3.5" />}
                  </button>
                )}
              </div>
              <div className="bg-[#FAF8F2] border border-[#EEE7DA] p-4 rounded-lg font-mono text-xs overflow-x-auto min-h-[120px] text-[#1F2421] relative">
                {executionResult?.sql ? (
                  <pre className="whitespace-pre-wrap break-all">{executionResult.sql}</pre>
                ) : (
                  <span className="text-[#8E9490] italic text-[11px] block mt-6 text-center">
                    SQL query will appear here once generated...
                  </span>
                )}
              </div>
            </div>

            {/* Syntax Validation report */}
            {executionResult && (
              <div className="space-y-2">
                <span className="text-[10px] uppercase font-semibold text-[#8E9490] tracking-wider block">Syntax Validation</span>
                {executionResult.is_valid_syntax ? (
                  <div className="border border-[#6E8B7E]/30 bg-[#6E8B7E]/5 text-[#6E8B7E] p-3.5 rounded text-xs font-medium">
                    ✓ AST Syntax verified & validated
                  </div>
                ) : (
                  <div className="border border-[#A0522D]/30 bg-[#A0522D]/5 text-[#A0522D] p-3.5 rounded text-xs">
                    <span className="font-semibold block mb-1">✕ Validation Error</span>
                    <p className="text-[10px] font-mono leading-relaxed">{executionResult.parsing_errors || 'Invalid query syntax.'}</p>
                  </div>
                )}
              </div>
            )}

            {/* Context tables used */}
            {executionResult?.retrieved_tables && executionResult.retrieved_tables.length > 0 && (
              <div className="space-y-2.5">
                <span className="text-[10px] uppercase font-semibold text-[#8E9490] tracking-wider block">Referenced Context Tables</span>
                <div className="space-y-1.5">
                  {executionResult.retrieved_tables.map((table: string) => (
                    <div key={table} className="flex items-center justify-between text-xs bg-[#FAF8F2] border border-[#EEE7DA] px-3 py-2 rounded">
                      <span className="font-mono text-[#5C625E] font-medium">{table}</span>
                      <span className="text-[10px] text-[#8E9490]">Active</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default Dashboard
