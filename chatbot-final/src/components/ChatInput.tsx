"use client"

import { useState, useRef, useEffect } from "react"
import { Plus, Mic, Paperclip, Send } from "lucide-react"
import type { ChatMessage } from "@/app/chat/page"

type ChatInputProps = {
  setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>
  onNewEntry?: (intent: string, question: string, session_id: string) => void
  setIsLoading?: (val: boolean) => void
  activeSessionId: string | null
}

type PendingUpload = {
  file: File
}

export default function ChatInput({
  setMessages,
  onNewEntry,
  setIsLoading,
  activeSessionId,
}: ChatInputProps) {
  const [input, setInput] = useState("")
  const [showMenu, setShowMenu] = useState(false)
  const [loading, setLoading] = useState(false)
  const [pendingUpload, setPendingUpload] = useState<PendingUpload | null>(null)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const menuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setShowMenu(false)
      }
    }
    document.addEventListener("mousedown", handleClickOutside)
    return () => document.removeEventListener("mousedown", handleClickOutside)
  }, [])

  function updateLoading(val: boolean) {
    setLoading(val)
    setIsLoading?.(val)
  }

  async function performUpload(file: File, userInput?: "replace" | "keep_both") {
    const token = localStorage.getItem("token")
    const formData = new FormData()
    formData.append("file", file)

    const url = userInput
      ? `http://localhost:8000/upload?user_input=${userInput}`
      : `http://localhost:8000/upload`

    updateLoading(true)

    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      })
      const data = await res.json()

      if (data.status === "conflict") {
        setPendingUpload({ file })
        return
      }

      setMessages(prev => [
        ...prev,
        {
          role: "assistant",
          text: "Document uploaded. Do you have any questions on documents?",
          source: "",
        },
      ])
    } catch (error) {
      console.error("Upload error:", error)
      setMessages(prev => [
        ...prev,
        { role: "bot", text: "Unable to upload the file. Please try again.", source: "" },
      ])
    } finally {
      updateLoading(false)
    }
  }

  async function upload(file: File) {
    setMessages(prev => [...prev, { role: "user", text: `📄 ${file.name}`, source: "" }])
    await performUpload(file)
  }

  async function handleReplace() {
    if (!pendingUpload) return
    const { file } = pendingUpload
    setPendingUpload(null)
    await performUpload(file, "replace")
  }

  async function handleKeepBoth() {
    if (!pendingUpload) return
    const { file } = pendingUpload
    setPendingUpload(null)
    await performUpload(file, "keep_both")
  }

  async function handleCancelUpload() {
    setPendingUpload(null)
    setMessages(prev => [
      ...prev,
      { role: "assistant", text: "Upload cancelled.", source: "" },
    ])
  }

  async function send() {
    if (!input.trim()) return
    const userMessage = input
    setInput("")

    setMessages(prev => [...prev, { role: "user", text: userMessage, source: "" }])
    updateLoading(true)

    try {
      const token = localStorage.getItem("token")
      const res = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          question: userMessage,
          session_id: activeSessionId ?? undefined,
        }),
      })

      const data = await res.json()
      const intent: string = data.intent ?? ""
      const session_id: string = data.session_id ?? activeSessionId ?? ""
      const fullText: string = data.Response ?? data.response ?? ""
      const sourceMatch = fullText.match(/\(Source:.*?\)/)
      const botText = fullText.replace(/\(Source:.*?\)/, "").trim()
      const sourceText = sourceMatch ? sourceMatch[0] : ""

      setMessages(prev => [...prev, { role: "bot", text: botText, source: sourceText }])

      if (intent && session_id) {
        onNewEntry?.(intent, userMessage, session_id)
      }
    } catch (error) {
      console.error("Error:", error)
      setMessages(prev => [
        ...prev,
        { role: "bot", text: "Something went wrong. Please try again.", source: "" },
      ])
    } finally {
      updateLoading(false)
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      if (!loading) send()
    }
  }

  return (
    <div className="border-t p-4 flex gap-3 relative">

      {pendingUpload && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-white dark:bg-[#2a2a2a] rounded-xl shadow-xl p-6 w-96 border dark:border-gray-700">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
              File already exists
            </h2>
            <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">
              <span className="font-medium text-gray-800 dark:text-gray-200">
                "{pendingUpload.file.name}"
              </span>{" "}
              already exists but the content has changed. What would you like to do?
            </p>
            <div className="flex flex-col gap-3">
              <button
                onClick={handleReplace}
                className="w-full px-4 py-2 bg-black dark:bg-white text-white dark:text-black rounded-lg text-sm font-medium hover:opacity-80 transition"
              >
                Replace — remove old version, keep new
              </button>
              <button
                onClick={handleKeepBoth}
                className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 text-gray-800 dark:text-gray-200 rounded-lg text-sm hover:bg-gray-100 dark:hover:bg-gray-700 transition"
              >
                Keep both versions
              </button>
              <button
                onClick={handleCancelUpload}
                className="w-full px-4 py-2 text-gray-400 dark:text-gray-500 text-sm hover:text-gray-600 dark:hover:text-gray-300 transition"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      <input
        type="file"
        ref={fileInputRef}
        className="hidden"
        onChange={e => {
          const file = e.target.files?.[0]
          if (file) upload(file)
          e.target.value = ""
        }}
      />

      <button
        onClick={e => { e.stopPropagation(); setShowMenu(!showMenu) }}
        disabled={loading}
        className="p-2 rounded-md bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600 transition disabled:opacity-50 disabled:cursor-not-allowed"
      >
        <Plus size={20} />
      </button>

      {showMenu && (
        <div
          ref={menuRef}
          className="absolute bottom-16 left-4 bg-white dark:bg-[#2a2a2a] shadow-lg rounded-lg p-2 w-52 border dark:border-gray-700"
        >
          <button
            onClick={() => { fileInputRef.current?.click(); setShowMenu(false) }}
            className="flex items-center gap-2 w-full px-3 py-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
          >
            <Paperclip size={16} />
            Add Files, Images & more...
          </button>
        </div>
      )}

      <input
        value={input}
        onChange={e => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
        className="flex-1 p-3 border rounded-md dark:bg-[#2a2a2a] disabled:opacity-60 disabled:cursor-not-allowed"
        placeholder={loading ? "Waiting for response..." : "Send message..."}
        disabled={loading}
      />

      <button className="p-2 rounded-md bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600 transition">
        <Mic size={20} />
      </button>

      <button
        onClick={send}
        disabled={loading || !input.trim()}
        className="px-6 bg-gray-100 dark:bg-gray-800 text-black dark:text-white rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
      >
        {loading ? (
          <span className="w-4 h-4 border-2 border-gray-400 border-t-black dark:border-t-white rounded-full animate-spin" />
        ) : (
          <Send size={20} />
        )}
        <span>{loading ? "Sending..." : "Send"}</span>
      </button>
    </div>
  )
}