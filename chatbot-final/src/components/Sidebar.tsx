"use client"

import { useState, useEffect, useRef } from "react"
import ThemeToggle from "./ThemeToggle"
import Logout from "./Logout"
import type { SidebarEntry } from "@/app/chat/page"

type SidebarProps = {
  entries: SidebarEntry[]
  activeSessionId: string | null
  createNewChat: () => void
  onSelectEntry: (entry: SidebarEntry) => Promise<void>
  onDeleteEntry: (entry: SidebarEntry) => Promise<void>
}

type ContextMenu = { x: number; y: number; entry: SidebarEntry } | null

export default function Sidebar({
  entries,
  activeSessionId,
  createNewChat,
  onSelectEntry,
  onDeleteEntry,
}: SidebarProps) {
  const [contextMenu, setContextMenu] = useState<ContextMenu>(null)
  const menuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setContextMenu(null)
      }
    }
    if (contextMenu) document.addEventListener("mousedown", handleClick)
    return () => document.removeEventListener("mousedown", handleClick)
  }, [contextMenu])

  function handleContextMenu(e: React.MouseEvent, entry: SidebarEntry) {
    e.preventDefault()
    e.stopPropagation()
    setContextMenu({ x: e.clientX, y: e.clientY, entry })
  }

  async function handleDelete() {
    if (!contextMenu) return
    const { entry } = contextMenu
    setContextMenu(null)
    await onDeleteEntry(entry)
  }

  return (
    <div className="w-[320px] border-r border-gray-200 dark:border-gray-800 p-4 flex flex-col select-none">
      <button
        onClick={createNewChat}
        className="w-full bg-gray-100 dark:bg-gray-800 text-black dark:text-white p-2 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 transition font-medium"
      >
        + New Chat
      </button>

      <div className="flex-1 overflow-y-auto mt-3 space-y-0.5 text-sm">
        {entries.length === 0 ? (
          <p className="text-gray-400 text-xs px-4 py-2">No history yet.</p>
        ) : (
          // Entries are already ordered newest-first from addEntry (prepend)
          entries.map((entry, index) => (
            <div
              key={`${entry.session_id}-${entry.question}-${index}`}
              onClick={() => onSelectEntry(entry)}
              onContextMenu={e => handleContextMenu(e, entry)}
              className={`px-4 py-2 rounded cursor-pointer truncate transition text-sm
                ${activeSessionId === entry.session_id
                  ? "bg-gray-200 dark:bg-gray-700 font-medium"
                  : "hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300"
                }`}
              title={`${entry.intent}\n"${entry.question}"`}
            >
              {entry.intent}
            </div>
          ))
        )}
      </div>

      {/* Right-click context menu */}
      {contextMenu && (
        <div
          ref={menuRef}
          style={{ top: contextMenu.y, left: contextMenu.x }}
          className="fixed z-50 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg shadow-xl py-1 min-w-[180px]"
        >
          <div className="px-4 py-2 text-xs text-gray-400 dark:text-gray-500 border-b border-gray-100 dark:border-gray-800 truncate max-w-[220px]">
            "{contextMenu.entry.question}"
          </div>
          <button
            onClick={handleDelete}
            className="w-full text-left px-4 py-2 text-sm text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 transition flex items-center gap-2"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4 flex-shrink-0"
              viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
              <polyline points="3 6 5 6 21 6" />
              <path d="M19 6l-1 14H6L5 6" />
              <path d="M10 11v6M14 11v6" />
              <path d="M9 6V4h6v2" />
            </svg>
            Delete this question
          </button>
        </div>
      )}

      <div className="mt-auto pt-12 w-full px-2">
        <div className="w-[200px] h-[60px] overflow-hidden">
          <ThemeToggle />
        </div>
        <div className="w-[200px] h-[60px] overflow-hidden">
          <Logout />
        </div>
      </div>
    </div>
  )
}
