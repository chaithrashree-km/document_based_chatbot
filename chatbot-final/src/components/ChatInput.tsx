"use client"

import { useState, useRef, useEffect } from "react"
import { Plus, Mic, Paperclip, Send } from "lucide-react"

export default function ChatInput({ setMessages, setIntents, setIsLoading }: any) {

  const [input, setInput] = useState("")
  const [showMenu, setShowMenu] = useState(false)
  const [loading, setLoading] = useState(false)

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

  async function upload(file: File) {
    const token = localStorage.getItem("token")
    const formData = new FormData()
    formData.append("file", file)

    updateLoading(true)

    try {
      const res = await fetch("http://localhost:8000/upload", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${token}`
        },
        body: formData
      })

      const data = await res.json()
      console.log("Upload response:", data)
      setMessages((prev: any) => [
        ...prev,
        { role: "assistant", text: `File uploaded and is being processed.`},
        { role: "assistant", text: `Do you have any questions on documents?`}
      ])
    } catch (error) {
      console.error("Upload error:", error)
      setMessages((prev: any) => [
        ...prev,
        { role: "bot", text: "Unable to upload the file. Please try again." }
      ])
    }finally {
      updateLoading(false)
    }
  }

  async function send() {
    if (!input) return
    const userMessage = input
    setInput("")
    setMessages((prev: any) => [
      ...prev,
      { role: "user", text: userMessage }
    ])

    updateLoading(true)

    try {
      const token = localStorage.getItem("token")
      const res = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`
        },
        body: JSON.stringify({ question: userMessage })
      })

      const data = await res.json()
      const intent=data.intent
      const fullText: string = data.Response ?? data.response ?? ""
      const sourceMatch = fullText.match(/\(Source:.*?\)/)
      const botText = fullText.replace(/\(Source:.*?\)/, "").trim()
      const sourceText = sourceMatch ? sourceMatch[0] : ""

      setMessages((prev: any) => [
        ...prev,
        { role: "bot", text: botText, source: sourceText }
      ])
       if (intent) {
          setIntents((prev: any) => [...prev, intent])
       }
      }catch (error) {
      console.error("Error:", error)
      setMessages((prev: any) => [
        ...prev,
        { role: "bot", text: "Something went wrong. Please try again." }
      ])
    }finally {
      updateLoading(false)
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  return (
    <div className="border-t p-4 flex gap-3 relative">

      <input
        type="file"
        ref={fileInputRef}
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) upload(file)
        }}
      />
      
      <button
        onClick={(e) => {
          e.stopPropagation()
          setShowMenu(!showMenu)
        }}
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
            onClick={() => {
              fileInputRef.current?.click()
              setShowMenu(false)
            }}
            className="flex items-center gap-2 w-full px-3 py-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
          >
            <Paperclip size={16} />
            Add Files,Images & more...
          </button>
        </div>
      )}

      <input
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={loading}
        className="flex-1 p-3 border rounded-md dark:bg-[#2a2a2a] disabled:opacity-60 disabled:cursor-not-allowed"
        placeholder={loading ? "Waiting for response..." : "Send message..."}
      />

      <button className="p-2 rounded-md bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600 transition">
        <Mic size={20} />
      </button>

      <button
        onClick={send}
        disabled={loading || !input.trim()}
        className="px-6 bg-gray-100 dark:bg-gray-800 text-black dark:text-white rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2">
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